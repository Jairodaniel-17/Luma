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
        }))
    }

    pub fn subscribe(&self) -> broadcast::Receiver<EventRecord> {
        self.0.sender.subscribe()
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
