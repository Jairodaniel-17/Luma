use crate::api::AppState;
use axum::extract::{Query, State};
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use futures_util::Stream;
use serde::Deserialize;
use std::convert::Infallible;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

#[derive(Debug, Deserialize)]
pub struct StreamQuery {
    pub since: Option<u64>,
    pub types: Option<String>,
    pub key_prefix: Option<String>,
    pub collection: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct EventsQuery {
    pub prefix: Option<String>,
    pub types: Option<String>,
    pub since: Option<u64>,
}

pub async fn events(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<EventsQuery>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    stream(
        State(state),
        headers,
        Query(StreamQuery {
            since: q.since,
            types: q.types,
            key_prefix: q.prefix,
            collection: None,
        }),
    )
    .await
}

pub async fn stream(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<StreamQuery>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let since = headers
        .get("last-event-id")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .or(q.since)
        .unwrap_or(0);

    let key_prefix = q.key_prefix.clone();
    let collection = q.collection.clone();
    let types: Option<Vec<String>> = q
        .types
        .as_deref()
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect());

    let metrics = state.engine.metrics();
    metrics.inc_sse_clients();
    let bus = state.engine.events().clone();
    let persist = state.engine.persist();

    let stream = async_stream::stream! {
        struct Guard(std::sync::Arc<crate::engine::Metrics>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.dec_sse_clients();
            }
        }
        let _guard = Guard(metrics);

        let mut last_sent_offset = since;

        if let Some(persist) = persist {
            let (tx, mut rx) = mpsc::unbounded_channel::<crate::engine::EventRecord>();
            let key_prefix2 = key_prefix.clone();
            let collection2 = collection.clone();
            let types2 = types.clone();
            tokio::task::spawn_blocking(move || {
                let _ = persist.for_each_event_since(since, |ev| {
                    if matches_filters(&ev, types2.as_ref(), key_prefix2.as_deref(), collection2.as_deref()) {
                        let _ = tx.send(ev);
                    }
                    true
                });
            });

            while let Some(ev) = rx.recv().await {
                last_sent_offset = ev.offset;
                yield Ok(to_sse(ev));
            }
        } else {
            for ev in bus.replay_since(since) {
                if !matches_filters(&ev, types.as_ref(), key_prefix.as_deref(), collection.as_deref()) {
                    continue;
                }
                last_sent_offset = ev.offset;
                yield Ok(to_sse(ev));
            }
        }

        let mut live = BroadcastStream::new(bus.subscribe());
        loop {
            match live.next().await {
                Some(Ok(ev)) => {
                    if matches_filters(&ev, types.as_ref(), key_prefix.as_deref(), collection.as_deref()) {
                        last_sent_offset = ev.offset;
                        yield Ok(to_sse(ev));
                    }
                }
                Some(Err(tokio_stream::wrappers::errors::BroadcastStreamRecvError::Lagged(n))) => {
                    let from_offset = last_sent_offset.saturating_add(1);
                    let to_offset = bus.last_published_offset().max(from_offset);
                    last_sent_offset = to_offset;
                    yield Ok(Event::default().event("gap").data(
                        serde_json::json!({
                            "from_offset": from_offset,
                            "to_offset": to_offset,
                            "dropped": n,
                        })
                        .to_string(),
                    ));
                }
                Some(Err(tokio_stream::wrappers::errors::BroadcastStreamRecvError::Closed)) | None => {
                    break;
                }
            }
        }
    };

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keepalive"),
    )
}

fn matches_filters(
    ev: &crate::engine::EventRecord,
    types: Option<&Vec<String>>,
    key_prefix: Option<&str>,
    collection: Option<&str>,
) -> bool {
    if let Some(types) = types {
        if !types.iter().any(|t| t == &ev.event_type) {
            return false;
        }
    }
    if let Some(prefix) = key_prefix {
        let Some(key) = ev.data.get("key").and_then(|v| v.as_str()) else {
            return false;
        };
        if !key.starts_with(prefix) {
            return false;
        }
    }
    if let Some(coll) = collection {
        let Some(c) = ev.data.get("collection").and_then(|v| v.as_str()) else {
            return false;
        };
        if c != coll {
            return false;
        }
    }
    true
}

fn to_sse(ev: crate::engine::EventRecord) -> Event {
    let data = serde_json::to_string(&ev).unwrap_or_else(|_| "{}".to_string());
    Event::default()
        .event(ev.event_type)
        .id(ev.offset.to_string())
        .data(data)
}
