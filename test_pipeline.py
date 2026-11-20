from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


def _sample_features() -> dict:
    feats = {f"V{i}": 0.0 for i in range(1, 29)}
    feats["Amount"] = 120.5
    return feats


def test_health_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "online"


def test_predict_does_not_learn_immediately_and_caches_features():
    before = client.get("/stats").json()
    tx_id = "unit-tx-001"

    pred = client.post("/predict", json={"features": _sample_features(), "transaction_id": tx_id})
    assert pred.status_code == 200
    body = pred.json()

    # Kiem tra schema va tach do tre tung mo hinh
    for key in ("fraud_probability", "action", "processing_time_ms", "lat_hat_ms", "lat_arf_ms", "lat_rf_ms"):
        assert key in body
    assert body["action"] in ("ALLOW", "BLOCK")

    # Khong duoc goi learn_one() trong /predict -> total giu nguyen, cache tang 1
    after_pred = client.get("/stats").json()
    assert after_pred["total"] == before["total"]
    assert after_pred["pending_cache_size"] == before["pending_cache_size"] + 1

    # Gui nhan tre cho dung tx_id -> total tang 1, cache quay ve ban dau
    fb = client.post("/feedback", json={"transaction_id": tx_id, "label": 1})
    assert fb.status_code == 200
    fb_body = fb.json()
    assert fb_body["status"] == "model_updated"
    assert set(fb_body["tradeoff_arf"].keys()) == {"precision", "recall", "fpr", "f1", "f2"}

    after_fb = client.get("/stats").json()
    assert after_fb["total"] == before["total"] + 1
    assert after_fb["fraud_detected"] == before["fraud_detected"] + 1
    assert after_fb["pending_cache_size"] == before["pending_cache_size"]
    assert after_fb["avg_latency_arf_ms"] >= 0.0


def test_feedback_rejects_unknown_transaction_id():
    resp = client.post("/feedback", json={"transaction_id": "non-existent-tx-999", "label": 0})
    assert resp.status_code == 400


def test_config_update_and_reload_reset():
    cfg_resp = client.post("/config", json={"block_threshold": 0.65, "window_size": 1000})
    assert cfg_resp.status_code == 200
    assert cfg_resp.json()["config"]["block_threshold"] == 0.65

    # Khoi phuc cau hinh chuan va kiem tra reload_models
    client.post("/config", json={"block_threshold": 0.70, "replay_interval": 10})
    rel = client.post("/reload_models")
    assert rel.status_code == 200
    assert rel.json()["status"] in ("models_reloaded", "files_not_found")

    drift_info = client.get("/drift_info").json()
    assert drift_info["drift_count"] == 0
    assert drift_info["drift_recovery_remaining"] == 0