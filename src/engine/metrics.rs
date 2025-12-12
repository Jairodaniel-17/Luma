use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default)]
pub struct Metrics {
    state_put_total: AtomicU64,
    state_delete_total: AtomicU64,
    vector_ops_total: AtomicU64,
    events_total: AtomicU64,
    sse_clients: AtomicU64,
}

impl Metrics {
    pub fn inc_state_put(&self) {
        self.state_put_total.fetch_add(1, Ordering::Relaxed);
    }
    pub fn inc_state_delete(&self) {
        self.state_delete_total.fetch_add(1, Ordering::Relaxed);
    }
    pub fn inc_vector_op(&self) {
        self.vector_ops_total.fetch_add(1, Ordering::Relaxed);
    }
    pub fn inc_events(&self) {
        self.events_total.fetch_add(1, Ordering::Relaxed);
    }
    pub fn inc_sse_clients(&self) {
        self.sse_clients.fetch_add(1, Ordering::Relaxed);
    }
    pub fn dec_sse_clients(&self) {
        self.sse_clients.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn render(&self) -> String {
        let state_put = self.state_put_total.load(Ordering::Relaxed);
        let state_delete = self.state_delete_total.load(Ordering::Relaxed);
        let vector_ops = self.vector_ops_total.load(Ordering::Relaxed);
        let events = self.events_total.load(Ordering::Relaxed);
        let sse_clients = self.sse_clients.load(Ordering::Relaxed);

        format!(
            concat!(
                "# TYPE state_put_total counter\n",
                "state_put_total {}\n",
                "# TYPE state_delete_total counter\n",
                "state_delete_total {}\n",
                "# TYPE vector_ops_total counter\n",
                "vector_ops_total {}\n",
                "# TYPE events_total counter\n",
                "events_total {}\n",
                "# TYPE sse_clients gauge\n",
                "sse_clients {}\n",
            ),
            state_put, state_delete, vector_ops, events, sse_clients
        )
    }
}
