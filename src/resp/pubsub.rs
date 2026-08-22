//! RESP Pub/Sub.
//!
//! F3.3 of `docs/PLAN-MAESTRO.md`. This is what Celery's fanout exchanges are
//! built on, so it is the last piece a kombu transport needs.
//!
//! ## Why a broker of its own
//!
//! The engine's `EventBus` carries every mutation as an event with a WAL offset
//! — durable, replayable, and shaped around state changes. Pub/Sub is the
//! opposite: fire-and-forget, no durability, no replay, and a message that
//! reaches nobody is simply gone. Mapping one onto the other would either give
//! Pub/Sub a durability guarantee Redis does not have (and clients would
//! eventually rely on it) or drag WAL-shaped concerns into a hot path that
//! should be a channel send.
//!
//! ## Tenant isolation
//!
//! A channel is addressed internally as `{tenant}\u{1f}{channel}`, so two
//! organizations publishing to `celeryev` never see each other. `PUBLISH`
//! therefore returns the number of receivers *in the caller's organization*,
//! not globally — a cross-tenant count would leak the existence of other
//! tenants' subscribers.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::mpsc;

/// Separator between tenant and channel. A unit separator cannot appear in a
/// channel name a client would type, so it cannot be used to escape a tenant.
const TENANT_SEP: char = '\u{1f}';

/// A message delivered to one subscriber.
#[derive(Clone, Debug, PartialEq)]
pub struct Delivery {
    /// The pattern that matched, for a `PSUBSCRIBE` delivery. `None` for a
    /// plain `SUBSCRIBE`, which changes the reply shape the client expects.
    pub pattern: Option<Vec<u8>>,
    pub channel: Vec<u8>,
    pub payload: Vec<u8>,
}

/// One connection's inbox.
pub struct Subscriber {
    pub id: u64,
    pub receiver: mpsc::Receiver<Delivery>,
}

struct Registry {
    /// Exact-channel subscriptions: internal channel name -> connection ids.
    channels: HashMap<String, Vec<u64>>,
    /// Pattern subscriptions: internal pattern -> connection ids.
    patterns: HashMap<String, Vec<u64>>,
    /// Where to deliver, by connection id.
    senders: HashMap<u64, mpsc::Sender<Delivery>>,
}

/// The Pub/Sub broker, shared by every connection on the listener.
pub struct PubSub {
    inner: Mutex<Registry>,
    next_id: AtomicU64,
}

impl Default for PubSub {
    fn default() -> Self {
        Self::new()
    }
}

impl PubSub {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Registry {
                channels: HashMap::new(),
                patterns: HashMap::new(),
                senders: HashMap::new(),
            }),
            next_id: AtomicU64::new(1),
        }
    }

    fn scoped(tenant: Option<&str>, name: &[u8]) -> String {
        let name = String::from_utf8_lossy(name);
        match tenant {
            Some(t) => format!("{t}{TENANT_SEP}{name}"),
            None => name.to_string(),
        }
    }

    /// Register a connection and hand back its inbox.
    ///
    /// The channel is bounded: a subscriber that stops reading must not let the
    /// publisher's memory grow without limit. When it fills, that subscriber's
    /// message is dropped — the same policy as the SSE stream, and the honest
    /// one for a protocol with no delivery guarantee.
    pub fn register(&self, capacity: usize) -> Subscriber {
        let (sender, receiver) = mpsc::channel(capacity.max(1));
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.inner.lock().senders.insert(id, sender);
        Subscriber { id, receiver }
    }

    /// Subscribe `id` to `channel`. Returns the connection's new subscription
    /// count, which is what the reply carries.
    pub fn subscribe(&self, id: u64, tenant: Option<&str>, channel: &[u8]) -> usize {
        let key = Self::scoped(tenant, channel);
        let mut registry = self.inner.lock();
        let entry = registry.channels.entry(key).or_default();
        if !entry.contains(&id) {
            entry.push(id);
        }
        Self::count_for(&registry, id)
    }

    pub fn psubscribe(&self, id: u64, tenant: Option<&str>, pattern: &[u8]) -> usize {
        let key = Self::scoped(tenant, pattern);
        let mut registry = self.inner.lock();
        let entry = registry.patterns.entry(key).or_default();
        if !entry.contains(&id) {
            entry.push(id);
        }
        Self::count_for(&registry, id)
    }

    pub fn unsubscribe(&self, id: u64, tenant: Option<&str>, channel: Option<&[u8]>) -> usize {
        let mut registry = self.inner.lock();
        match channel {
            Some(name) => {
                let key = Self::scoped(tenant, name);
                if let Some(subs) = registry.channels.get_mut(&key) {
                    subs.retain(|s| *s != id);
                    if subs.is_empty() {
                        registry.channels.remove(&key);
                    }
                }
            }
            // No channel means "all of them", which is what a bare UNSUBSCRIBE
            // does in Redis.
            None => {
                registry.channels.retain(|_, subs| {
                    subs.retain(|s| *s != id);
                    !subs.is_empty()
                });
            }
        }
        Self::count_for(&registry, id)
    }

    pub fn punsubscribe(&self, id: u64, tenant: Option<&str>, pattern: Option<&[u8]>) -> usize {
        let mut registry = self.inner.lock();
        match pattern {
            Some(name) => {
                let key = Self::scoped(tenant, name);
                if let Some(subs) = registry.patterns.get_mut(&key) {
                    subs.retain(|s| *s != id);
                    if subs.is_empty() {
                        registry.patterns.remove(&key);
                    }
                }
            }
            None => {
                registry.patterns.retain(|_, subs| {
                    subs.retain(|s| *s != id);
                    !subs.is_empty()
                });
            }
        }
        Self::count_for(&registry, id)
    }

    /// Drop a connection entirely. Called when the socket closes, and what
    /// keeps the registry from growing one dead entry per disconnect.
    pub fn drop_connection(&self, id: u64) {
        let mut registry = self.inner.lock();
        registry.senders.remove(&id);
        registry.channels.retain(|_, subs| {
            subs.retain(|s| *s != id);
            !subs.is_empty()
        });
        registry.patterns.retain(|_, subs| {
            subs.retain(|s| *s != id);
            !subs.is_empty()
        });
    }

    /// Publish to a channel. Returns how many subscribers received it.
    ///
    /// A connection subscribed both directly and by a matching pattern is
    /// counted once per delivery, as Redis does — it genuinely receives two
    /// messages.
    pub fn publish(&self, tenant: Option<&str>, channel: &[u8], payload: &[u8]) -> usize {
        let key = Self::scoped(tenant, channel);
        let registry = self.inner.lock();

        let mut deliveries: Vec<(u64, Delivery)> = Vec::new();
        if let Some(subs) = registry.channels.get(&key) {
            for id in subs {
                deliveries.push((
                    *id,
                    Delivery {
                        pattern: None,
                        channel: channel.to_vec(),
                        payload: payload.to_vec(),
                    },
                ));
            }
        }
        for (pattern_key, subs) in &registry.patterns {
            // Match against the scoped names on both sides, so a pattern can
            // never reach across tenants even if it is `*`.
            if !crate::resp::commands::glob_match(pattern_key.as_bytes(), key.as_bytes()) {
                continue;
            }
            let bare_pattern = pattern_key
                .split_once(TENANT_SEP)
                .map(|(_, p)| p)
                .unwrap_or(pattern_key);
            for id in subs {
                deliveries.push((
                    *id,
                    Delivery {
                        pattern: Some(bare_pattern.as_bytes().to_vec()),
                        channel: channel.to_vec(),
                        payload: payload.to_vec(),
                    },
                ));
            }
        }

        let mut delivered = 0;
        for (id, message) in deliveries {
            if let Some(sender) = registry.senders.get(&id) {
                // try_send, never await: a publisher must not be stalled by one
                // slow subscriber. A full inbox drops that message, which is
                // what "no delivery guarantee" means in practice.
                if sender.try_send(message).is_ok() {
                    delivered += 1;
                }
            }
        }
        delivered
    }

    /// Channels with at least one subscriber in this tenant, for `PUBSUB
    /// CHANNELS`.
    pub fn channels(&self, tenant: Option<&str>, pattern: Option<&[u8]>) -> Vec<Vec<u8>> {
        let registry = self.inner.lock();
        let prefix = tenant.map(|t| format!("{t}{TENANT_SEP}"));
        registry
            .channels
            .keys()
            .filter_map(|key| match &prefix {
                Some(p) => key.strip_prefix(p.as_str()).map(|c| c.to_string()),
                // A platform connection sees only unscoped channels, never
                // another tenant's.
                None => (!key.contains(TENANT_SEP)).then(|| key.clone()),
            })
            .filter(|channel| match pattern {
                Some(p) => crate::resp::commands::glob_match(p, channel.as_bytes()),
                None => true,
            })
            .map(|c| c.into_bytes())
            .collect()
    }

    /// Subscriber count for one channel, for `PUBSUB NUMSUB`.
    pub fn subscriber_count(&self, tenant: Option<&str>, channel: &[u8]) -> usize {
        let key = Self::scoped(tenant, channel);
        self.inner
            .lock()
            .channels
            .get(&key)
            .map(|subs| subs.len())
            .unwrap_or(0)
    }

    /// Total registered connections plus subscription entries. A test aid, and
    /// the only way to distinguish "delivery fails because the receiver is
    /// gone" from "the registration was actually removed".
    pub fn tracked(&self) -> usize {
        let registry = self.inner.lock();
        registry.senders.len() + registry.channels.len() + registry.patterns.len()
    }

    fn count_for(registry: &Registry, id: u64) -> usize {
        let channels = registry
            .channels
            .values()
            .filter(|subs| subs.contains(&id))
            .count();
        let patterns = registry
            .patterns
            .values()
            .filter(|subs| subs.contains(&id))
            .count();
        channels + patterns
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn broker() -> Arc<PubSub> {
        Arc::new(PubSub::new())
    }

    #[tokio::test]
    async fn a_message_reaches_every_subscriber() {
        // Fanout is the point: Celery's fanout exchange delivers to all workers.
        let ps = broker();
        let mut a = ps.register(16);
        let mut b = ps.register(16);
        ps.subscribe(a.id, None, b"news");
        ps.subscribe(b.id, None, b"news");

        assert_eq!(ps.publish(None, b"news", b"hello"), 2);
        assert_eq!(a.receiver.recv().await.unwrap().payload, b"hello");
        assert_eq!(b.receiver.recv().await.unwrap().payload, b"hello");
    }

    #[tokio::test]
    async fn publishing_to_a_channel_with_no_subscribers_delivers_to_none() {
        // Fire and forget: the message is simply gone, and the count says so.
        let ps = broker();
        assert_eq!(ps.publish(None, b"empty", b"x"), 0);
    }

    #[tokio::test]
    async fn a_pattern_subscriber_receives_and_learns_the_pattern() {
        // The reply shape differs for a pattern delivery, so the subscriber has
        // to know which pattern matched.
        let ps = broker();
        let mut sub = ps.register(16);
        ps.psubscribe(sub.id, None, b"news.*");

        assert_eq!(ps.publish(None, b"news.sport", b"goal"), 1);
        let message = sub.receiver.recv().await.unwrap();
        assert_eq!(message.pattern.as_deref(), Some(b"news.*".as_slice()));
        assert_eq!(message.channel, b"news.sport");
    }

    #[tokio::test]
    async fn a_pattern_that_does_not_match_receives_nothing() {
        let ps = broker();
        let sub = ps.register(16);
        ps.psubscribe(sub.id, None, b"news.*");
        assert_eq!(ps.publish(None, b"sports.football", b"x"), 0);
    }

    #[tokio::test]
    async fn a_double_subscriber_gets_both_copies() {
        // Subscribed directly *and* by pattern: Redis delivers twice, and a
        // client that deduplicated would drop legitimate messages.
        let ps = broker();
        let mut sub = ps.register(16);
        ps.subscribe(sub.id, None, b"news");
        ps.psubscribe(sub.id, None, b"n*");

        assert_eq!(ps.publish(None, b"news", b"x"), 2);
        assert!(sub.receiver.recv().await.is_some());
        assert!(sub.receiver.recv().await.is_some());
    }

    #[tokio::test]
    async fn tenants_cannot_see_each_others_channels() {
        // Two organizations both publishing to `celeryev` is the realistic case,
        // and crossing it would leak one org's events into another's workers.
        let ps = broker();
        let mut acme = ps.register(16);
        let mut globex = ps.register(16);
        ps.subscribe(acme.id, Some("acme"), b"celeryev");
        ps.subscribe(globex.id, Some("globex"), b"celeryev");

        assert_eq!(ps.publish(Some("acme"), b"celeryev", b"acme-event"), 1);
        assert_eq!(acme.receiver.recv().await.unwrap().payload, b"acme-event");
        assert!(globex.receiver.try_recv().is_err());
    }

    #[tokio::test]
    async fn a_wildcard_pattern_cannot_cross_tenants() {
        // `PSUBSCRIBE *` is the obvious attempt; matching on the scoped names is
        // what stops it.
        let ps = broker();
        let globex = ps.register(16);
        ps.psubscribe(globex.id, Some("globex"), b"*");
        assert_eq!(ps.publish(Some("acme"), b"secret", b"payload"), 0);
    }

    #[tokio::test]
    async fn unsubscribe_stops_delivery_and_reports_the_remaining_count() {
        let ps = broker();
        let sub = ps.register(16);
        assert_eq!(ps.subscribe(sub.id, None, b"a"), 1);
        assert_eq!(ps.subscribe(sub.id, None, b"b"), 2);
        assert_eq!(ps.unsubscribe(sub.id, None, Some(b"a")), 1);
        assert_eq!(ps.publish(None, b"a", b"x"), 0);
        assert_eq!(ps.publish(None, b"b", b"x"), 1);
    }

    #[tokio::test]
    async fn a_bare_unsubscribe_clears_everything() {
        let ps = broker();
        let sub = ps.register(16);
        ps.subscribe(sub.id, None, b"a");
        ps.subscribe(sub.id, None, b"b");
        assert_eq!(ps.unsubscribe(sub.id, None, None), 0);
        assert_eq!(ps.publish(None, b"a", b"x"), 0);
    }

    #[tokio::test]
    async fn subscribing_twice_to_one_channel_is_idempotent() {
        let ps = broker();
        let sub = ps.register(16);
        assert_eq!(ps.subscribe(sub.id, None, b"a"), 1);
        assert_eq!(ps.subscribe(sub.id, None, b"a"), 1);
        assert_eq!(ps.publish(None, b"a", b"x"), 1, "one delivery, not two");
    }

    #[tokio::test]
    async fn dropping_a_connection_removes_it_everywhere() {
        // Without this the registry grows one dead entry per disconnect, and a
        // long-lived server leaks steadily.
        let ps = broker();
        let sub = ps.register(16);
        ps.subscribe(sub.id, None, b"a");
        ps.psubscribe(sub.id, None, b"b*");
        assert!(ps.tracked() > 0);
        ps.drop_connection(sub.id);

        assert_eq!(ps.publish(None, b"a", b"x"), 0);
        assert_eq!(ps.publish(None, b"bee", b"x"), 0);
        assert!(ps.channels(None, None).is_empty());
        assert_eq!(
            ps.tracked(),
            0,
            "the registration itself must be removed, not merely left              undeliverable"
        );
    }

    #[tokio::test]
    async fn a_full_inbox_drops_the_message_rather_than_stalling_the_publisher() {
        // One subscriber that stopped reading must not be able to block every
        // publisher — the same policy the SSE stream uses.
        let ps = broker();
        let sub = ps.register(1);
        ps.subscribe(sub.id, None, b"busy");

        assert_eq!(ps.publish(None, b"busy", b"first"), 1);
        // The inbox holds one; the next is dropped rather than awaited.
        assert_eq!(ps.publish(None, b"busy", b"second"), 0);
    }

    #[tokio::test]
    async fn pubsub_channels_and_numsub_are_tenant_scoped() {
        let ps = broker();
        let acme = ps.register(16);
        let globex = ps.register(16);
        ps.subscribe(acme.id, Some("acme"), b"jobs");
        ps.subscribe(globex.id, Some("globex"), b"jobs");
        ps.subscribe(globex.id, Some("globex"), b"other");

        assert_eq!(ps.channels(Some("acme"), None), vec![b"jobs".to_vec()]);
        let mut globex_channels = ps.channels(Some("globex"), None);
        globex_channels.sort();
        assert_eq!(globex_channels, vec![b"jobs".to_vec(), b"other".to_vec()]);
        assert_eq!(ps.subscriber_count(Some("acme"), b"jobs"), 1);
        assert_eq!(ps.subscriber_count(Some("acme"), b"other"), 0);
    }

    #[tokio::test]
    async fn pubsub_channels_honours_a_pattern() {
        let ps = broker();
        let sub = ps.register(16);
        ps.subscribe(sub.id, None, b"news.a");
        ps.subscribe(sub.id, None, b"jobs.b");
        assert_eq!(ps.channels(None, Some(b"news.*")), vec![b"news.a".to_vec()]);
    }

    #[tokio::test]
    async fn a_binary_payload_survives() {
        // Celery event bodies are not text.
        let ps = broker();
        let mut sub = ps.register(16);
        ps.subscribe(sub.id, None, b"bin");
        let payload = vec![0x00u8, 0xFF, 0x80];
        ps.publish(None, b"bin", &payload);
        assert_eq!(sub.receiver.recv().await.unwrap().payload, payload);
    }
}
