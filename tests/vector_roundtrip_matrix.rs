use luma::config::Config;
use luma::engine::Engine;
use luma::vector::{Metric, SearchOptions, SearchRequest, VectorItem};
use tokio_util::sync::CancellationToken;

fn config_for(dir: &str) -> Config {
    Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "test".to_string(),
        data_dir: Some(dir.to_string()),
        snapshot_interval_secs: 3600,
        ..Config::default()
    }
}

#[tokio::test]
async fn vector_roundtrip_matrix_dims() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();
    let config = config_for(&data_dir);
    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();

    let dims = [8usize, 128, 384, 768];
    for &dim in &dims {
        let collection = format!("roundtrip_{dim}");
        engine
            .create_vector_collection(&collection, dim, Metric::Cosine)
            .unwrap();
        for idx in 0..32usize {
            let mut vector = vec![0.0f32; dim];
            vector[idx % dim] = 1.0;
            engine
                .vector_upsert(
                    &collection,
                    &format!("id-{idx}"),
                    VectorItem {
                        vector,
                        meta: serde_json::json!({ "dim": dim, "idx": idx }),
                        mmap_offset: None,
                    },
                )
                .unwrap();
        }
    }
    drop(engine);

    let reopened = Engine::new(config, CancellationToken::new()).unwrap();
    for &dim in &dims {
        let collection = format!("roundtrip_{dim}");
        let hits = reopened
            .vector_search(
                &collection,
                SearchRequest {
                    vector: vec![1.0; dim],
                    k: 5,
                    options: SearchOptions {
                        filters: None,
                        filter: None,
                        min_score: None,
                        include_meta: true,
                        allowed_ids: None,
                    },
                },
            )
            .unwrap();
        assert!(
            !hits.is_empty(),
            "expected hits for dim {dim} after restart"
        );
        assert!(
            hits.iter().all(|hit| hit.meta.as_ref().is_some()),
            "expected metadata for dim {dim}"
        );
    }
}
