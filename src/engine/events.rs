use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::broadcast;

#[derive(Clone)]
pub struct EventBus(Arc<Inner>);

struct Inner {
    sender: broadcast::Sender<EventRecord>,
    buffer: Mutex<VecDeque<EventRecord>>,
    next_offset: AtomicU64,
    last_published_offset: AtomicU64,
    capacity: usize,
    /// Serializes offset allocation with WAL append so that offset order == file
    /// order. Callers hold this across `next_record` + WAL append + `publish_record`
    /// so two concurrent writers can't append a higher offset before a lower one.
    append_lock: Mutex<()>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventRecord {
    pub offset: u64,
    pub ts_ms: u64,
    #[serde(rename = "type")]
    pub event_type: String,
    pub data: serde_json::Value,
}

impl EventBus {
    pub fn new(capacity: usize, live_broadcast_capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(live_broadcast_capacity.max(16));
        Self(Arc::new(Inner {
            sender,
            buffer: Mutex::new(VecDeque::with_capacity(capacity.min(1024))),
            next_offset: AtomicU64::new(1),
            last_published_offset: AtomicU64::new(0),
            capacity,
            append_lock: Mutex::new(()),
        }))
    }

    pub fn subscribe(&self) -> broadcast::Receiver<EventRecord> {
        self.0.sender.subscribe()
    }

    /// Acquire the append serialization lock. The caller MUST hold the returned
    /// guard across `next_record` (offset allocation) AND the subsequent WAL
    /// append AND `publish_record`, so that the order offsets are assigned in is
    /// exactly the order records are appended to the WAL and published. Without
    /// this, two concurrent writers could append a higher offset before a lower
    /// one, producing false gaps on replay and a non-monotonic
    /// `last_published_offset`.
    ///
    /// Lock order: this guard is the outermost engine write lock; it is always
    /// taken before `Persist`'s internal `wal_lock`, never the reverse, so it
    /// cannot deadlock with WAL IO.
    pub fn append_guard(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.0.append_lock.lock()
    }

    pub fn next_record(
        &self,
        event_type: impl Into<String>,
        data: serde_json::Value,
    ) -> EventRecord {
        let offset = self.0.next_offset.fetch_add(1, Ordering::Relaxed);
        EventRecord {
            offset,
            ts_ms: now_ms(),
            event_type: event_type.into(),
            data,
        }
    }

    pub fn publish_record(&self, record: EventRecord) {
        self.0
            .last_published_offset
            .store(record.offset, Ordering::Relaxed);
        {
            let mut buf = self.0.buffer.lock();
            buf.push_back(record.clone());
            while buf.len() > self.0.capacity {
                buf.pop_front();
            }
        }
        let _ = self.0.sender.send(record);
    }

    pub fn replay_since(&self, last_offset: u64) -> Vec<EventRecord> {
        let buf = self.0.buffer.lock();
        buf.iter()
            .filter(|e| e.offset > last_offset)
            .cloned()
            .collect()
    }

    pub fn replay_since_with_gap(
        &self,
        last_offset: u64,
    ) -> (Vec<EventRecord>, Option<(u64, u64)>) {
        let buf = self.0.buffer.lock();
        let earliest = buf.front().map(|event| event.offset);
        let gap = earliest.and_then(|earliest| {
            let expected = last_offset.saturating_add(1);
            (expected < earliest).then_some((expected, earliest.saturating_sub(1)))
        });
        let events = buf
            .iter()
            .filter(|e| e.offset > last_offset)
            .cloned()
            .collect();
        (events, gap)
    }

    pub fn last_published_offset(&self) -> u64 {
        self.0.last_published_offset.load(Ordering::Relaxed)
    }

    pub fn set_next_offset(&self, next: u64) {
        self.0.next_offset.store(next.max(1), Ordering::Relaxed);
    }
}

fn now_ms() -> u64 {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    dur.as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::{EventBus, EventRecord};
    use serde_json::json;

    #[test]
    fn replay_since_with_gap_reports_missing_prefix() {
        let bus = EventBus::new(2, 16);
        for offset in 1..=4u64 {
            bus.publish_record(EventRecord {
                offset,
                ts_ms: 0,
                event_type: "state_updated".to_string(),
                data: json!({"key": format!("k{offset}")}),
            });
        }

        let (events, gap) = bus.replay_since_with_gap(1);
        assert_eq!(gap, Some((2, 2)));
        assert_eq!(
            events.iter().map(|event| event.offset).collect::<Vec<_>>(),
            vec![3, 4]
        );
    }

    #[test]
    fn append_guard_keeps_offset_order_equal_to_append_order() {
        use std::sync::{Arc, Mutex};

        let bus = Arc::new(EventBus::new(1024, 16));
        // Shared "append log": offsets are pushed in the order the records are
        // "written", while holding the append guard. If allocation and append
        // are serialized together this must be strictly increasing.
        let append_log = Arc::new(Mutex::new(Vec::<u64>::new()));

        let mut handles = Vec::new();
        for _ in 0..8 {
            let bus = Arc::clone(&bus);
            let append_log = Arc::clone(&append_log);
            handles.push(std::thread::spawn(move || {
                for _ in 0..100 {
                    let _guard = bus.append_guard();
                    let record = bus.next_record("state_updated", json!({}));
                    // Simulate the WAL append happening under the same guard.
                    append_log.lock().unwrap().push(record.offset);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let log = append_log.lock().unwrap();
        assert_eq!(log.len(), 800);
        assert!(
            log.windows(2).all(|w| w[1] == w[0] + 1),
            "offsets must be contiguous and appended in allocation order"
        );
    }
}
