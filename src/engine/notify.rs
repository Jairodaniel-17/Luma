//! Per-key wakeups for blocking reads.
//!
//! F0.3 of `docs/PLAN-MAESTRO.md`. This is the primitive the RESP blocking
//! commands are built on: `BLPOP` and friends have to park until *some* writer
//! touches *one of several* keys, then wake exactly one parked reader.
//!
//! ## The two properties that matter
//!
//! **Wake one, not all.** A push that appended a single element must not wake
//! fifty waiters to have forty-nine discover an empty list and park again. That
//! is why the entry holds a [`Notify`] and the writer calls `notify_one`.
//!
//! **No lost wakeups.** A waiter has to register its interest *before* it
//! re-checks whether data is available, or a push landing in between is missed
//! and the waiter sleeps through data that is already there. [`Waiter`] exists
//! to make that ordering the only way to use this: you register, then re-check,
//! then await.
//!
//! ```ignore
//! let waiter = notifier.waiter(&keys);   // registered from here on
//! if let Some(item) = try_pop(&keys) {   // re-check *after* registering
//!     return Some(item);
//! }
//! waiter.wait(timeout).await;            // a push in between is not lost
//! ```

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;

struct Entry {
    notify: Notify,
    /// Live waiters. The map entry is dropped when this hits zero, so a server
    /// that has served a million distinct keys does not keep a million
    /// notifiers alive.
    waiters: AtomicUsize,
}

/// Registry of per-key wakeup channels.
#[derive(Default)]
pub struct KeyNotifier {
    keys: Mutex<HashMap<String, Arc<Entry>>>,
}

impl KeyNotifier {
    pub fn new() -> Self {
        Self::default()
    }

    /// Wake at most one waiter on `key`.
    ///
    /// Does nothing when nobody is parked: with no entry there is no waiter to
    /// wake, and storing a permit for a future waiter would be wrong — it would
    /// let a later `BLPOP` return immediately on a push it did not observe,
    /// against data that another reader has already taken.
    pub fn notify_one(&self, key: &str) {
        let entry = self.keys.lock().get(key).cloned();
        if let Some(entry) = entry {
            entry.notify.notify_one();
        }
    }

    /// Wake at most one waiter on each of `keys`.
    pub fn notify_each(&self, keys: &[String]) {
        for key in keys {
            self.notify_one(key);
        }
    }

    /// Register interest in `keys` and return the guard to await on.
    ///
    /// Registration happens here, synchronously, so the caller can re-check its
    /// data source afterwards without a window for a lost wakeup.
    pub fn waiter(self: &Arc<Self>, keys: &[String]) -> Waiter {
        let mut registered = Vec::with_capacity(keys.len());
        {
            let mut map = self.keys.lock();
            for key in keys {
                let entry = map
                    .entry(key.clone())
                    .or_insert_with(|| {
                        Arc::new(Entry {
                            notify: Notify::new(),
                            waiters: AtomicUsize::new(0),
                        })
                    })
                    .clone();
                entry.waiters.fetch_add(1, Ordering::AcqRel);
                registered.push((key.clone(), entry));
            }
        }
        Waiter {
            notifier: Arc::clone(self),
            registered,
        }
    }

    /// Number of keys currently holding a notifier. Test and metrics aid: it
    /// must return to zero once every waiter is gone.
    pub fn tracked_keys(&self) -> usize {
        self.keys.lock().len()
    }
}

/// A registered interest in one or more keys.
///
/// Dropping it deregisters, which is what keeps the registry from growing
/// without bound — including when a client disconnects mid-`BLPOP`, since the
/// future holding this is dropped with the connection.
pub struct Waiter {
    notifier: Arc<KeyNotifier>,
    registered: Vec<(String, Arc<Entry>)>,
}

impl Waiter {
    /// Park until one of the registered keys is notified, or `timeout` elapses.
    ///
    /// Returns the key that fired, or `None` on timeout. `None` for `timeout`
    /// means wait forever — Redis spells that `timeout 0`, and the translation
    /// belongs at the command layer, not here.
    pub async fn wait(&self, timeout: Option<Duration>) -> Option<String> {
        // Build every `Notified` future before awaiting any of them: creating
        // the future is what enlists this task, so enlisting on key B only
        // after key A resolves would drop a wakeup on B.
        let mut futures: Vec<_> = self
            .registered
            .iter()
            .map(|(key, entry)| {
                let notified = entry.notify.notified();
                (key.clone(), Box::pin(notified))
            })
            .collect();

        let first_fired = async {
            loop {
                // `select_all` needs at least one future; an empty key list can
                // only wait for the timeout.
                if futures.is_empty() {
                    std::future::pending::<()>().await;
                }
                let (_, index, _) = futures_util::future::select_all(
                    futures.iter_mut().map(|(_, fut)| fut.as_mut()),
                )
                .await;
                return futures[index].0.clone();
            }
        };

        match timeout {
            Some(limit) => tokio::time::timeout(limit, first_fired).await.ok(),
            None => Some(first_fired.await),
        }
    }
}

impl Drop for Waiter {
    fn drop(&mut self) {
        let mut map = self.notifier.keys.lock();
        for (key, entry) in &self.registered {
            // Remove the entry only when this was the last waiter *and* the map
            // still points at this exact entry — a new waiter may have created
            // a fresh one for the same key in between.
            if entry.waiters.fetch_sub(1, Ordering::AcqRel) == 1 {
                if let Some(current) = map.get(key) {
                    if Arc::ptr_eq(current, entry) && current.waiters.load(Ordering::Acquire) == 0 {
                        map.remove(key);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn keys(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    #[tokio::test]
    async fn a_push_wakes_exactly_one_of_many_waiters() {
        // The thundering-herd property. Fifty parked readers, one element
        // pushed: waking all fifty would have forty-nine spin and re-park.
        let notifier = Arc::new(KeyNotifier::new());
        let woken = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();

        for _ in 0..50 {
            let notifier = Arc::clone(&notifier);
            let woken = Arc::clone(&woken);
            handles.push(tokio::spawn(async move {
                let waiter = notifier.waiter(&keys(&["jobs"]));
                if waiter
                    .wait(Some(Duration::from_millis(500)))
                    .await
                    .is_some()
                {
                    woken.fetch_add(1, Ordering::AcqRel);
                }
            }));
        }

        // Give every task time to register before the single push.
        while notifier.tracked_keys() == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
        notifier.notify_one("jobs");

        for handle in handles {
            let _ = handle.await;
        }
        assert_eq!(
            woken.load(Ordering::Acquire),
            1,
            "exactly one waiter must wake per notify_one"
        );
    }

    #[tokio::test]
    async fn wait_reports_which_key_fired() {
        let notifier = Arc::new(KeyNotifier::new());
        let waiter = notifier.waiter(&keys(&["a", "b", "c"]));
        let notifier2 = Arc::clone(&notifier);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            notifier2.notify_one("b");
        });
        assert_eq!(
            waiter.wait(Some(Duration::from_secs(2))).await.as_deref(),
            Some("b"),
            "a multi-key waiter must say which key woke it, since BLPOP has to \
             pop from that one"
        );
    }

    #[tokio::test]
    async fn timeout_returns_none_without_hanging() {
        let notifier = Arc::new(KeyNotifier::new());
        let waiter = notifier.waiter(&keys(&["idle"]));
        let start = std::time::Instant::now();
        assert!(waiter.wait(Some(Duration::from_millis(80))).await.is_none());
        let elapsed = start.elapsed();
        assert!(
            elapsed >= Duration::from_millis(70),
            "returned before the timeout: {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "overshot the timeout badly: {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn notify_with_nobody_waiting_is_not_stored() {
        // A stored permit would let a later waiter return immediately for a push
        // it never observed — and whose element another reader has since taken.
        let notifier = Arc::new(KeyNotifier::new());
        notifier.notify_one("ghost");
        assert_eq!(notifier.tracked_keys(), 0);

        let waiter = notifier.waiter(&keys(&["ghost"]));
        assert!(
            waiter.wait(Some(Duration::from_millis(80))).await.is_none(),
            "a notify that had no waiter must not be replayed to a later one"
        );
    }

    #[tokio::test]
    async fn registry_returns_to_empty_when_waiters_go_away() {
        let notifier = Arc::new(KeyNotifier::new());
        {
            let _a = notifier.waiter(&keys(&["k1", "k2"]));
            let _b = notifier.waiter(&keys(&["k2", "k3"]));
            assert_eq!(notifier.tracked_keys(), 3);
        }
        assert_eq!(
            notifier.tracked_keys(),
            0,
            "dropping every waiter must free the notifiers — a client that \
             disconnects mid-BLPOP drops its future, and a server that has seen \
             many keys must not accumulate one entry per key forever"
        );
    }

    #[tokio::test]
    async fn two_notifications_serve_two_parked_waiters() {
        // Redis semantics: two blocked BLPOPs and two pushes means both clients
        // get served. Both waiters are actually parked before the notifications
        // land, which is the real sequence — a `Notify` holds at most one
        // permit, so notifying before anyone awaits can only release one.
        let notifier = Arc::new(KeyNotifier::new());
        let woken = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..2 {
            let notifier = Arc::clone(&notifier);
            let woken = Arc::clone(&woken);
            handles.push(tokio::spawn(async move {
                let waiter = notifier.waiter(&keys(&["shared"]));
                if waiter
                    .wait(Some(Duration::from_millis(500)))
                    .await
                    .is_some()
                {
                    woken.fetch_add(1, Ordering::AcqRel);
                }
            }));
        }
        while notifier.tracked_keys() == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;

        notifier.notify_one("shared");
        notifier.notify_one("shared");
        for handle in handles {
            let _ = handle.await;
        }
        assert_eq!(woken.load(Ordering::Acquire), 2);
    }

    #[tokio::test]
    async fn a_push_between_registering_and_awaiting_is_not_lost() {
        // The lost-wakeup guard, and the reason `waiter()` is separate from
        // `wait()`: a writer that lands in that gap must still release the
        // waiter rather than leaving it parked on data that is already there.
        let notifier = Arc::new(KeyNotifier::new());
        let waiter = notifier.waiter(&keys(&["race"]));
        notifier.notify_one("race");
        assert_eq!(
            waiter
                .wait(Some(Duration::from_millis(200)))
                .await
                .as_deref(),
            Some("race")
        );
    }

    #[tokio::test]
    async fn an_empty_key_list_only_times_out() {
        // Degenerate but reachable: a command whose key list was filtered to
        // nothing must behave as a plain sleep, not panic in select_all.
        let notifier = Arc::new(KeyNotifier::new());
        let waiter = notifier.waiter(&[]);
        assert!(waiter.wait(Some(Duration::from_millis(50))).await.is_none());
    }
}
