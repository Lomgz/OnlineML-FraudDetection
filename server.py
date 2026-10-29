from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import threading
import time
import os
import pickle

from river import linear_model, preprocessing, metrics, compose, optim
from sklearn.metrics import average_precision_score, precision_recall_curve, auc

app = FastAPI(
    title="Payment Edge & Streaming Broker Fraud Detection Engine",
    description="Online Machine Learning engine designed for Payment Edge Gateways and Cloud Message Brokers (Kafka/Flink)",
    version="2.0.0"
)

MODEL_SGD_FILE = "model_sgd.pkl"
MODEL_ADAM_FILE = "model_adam.pkl"
MODEL_RF_FILE = "model_rf.pkl"

if os.path.exists(MODEL_SGD_FILE) and os.path.exists(MODEL_ADAM_FILE) and os.path.exists(MODEL_RF_FILE): 
    print("[Server] Successfully loaded pretrained models from .pkl files!")
    with open(MODEL_SGD_FILE, "rb") as f:
        model_sgd = pickle.load(f)
    with open(MODEL_ADAM_FILE, "rb") as f:
        model_adam = pickle.load(f)
    with open(MODEL_RF_FILE, "rb") as f:
        model_rf = pickle.load(f)
else:
    print("[Server] .pkl files not found. Starting with empty models...")
    opt_sgd = optim.SGD(lr = 0.1)
    model_sgd = compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(optimizer = opt_sgd)
    )
    opt_adam = optim.Adam(lr = 0.01)
    model_adam = compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(optimizer = opt_adam)
    )
    model_rf = None

# Evaluation Protocol v2 Metrics: PrecisionRecallAUC & Pr@k
if not hasattr(metrics, "PrecisionRecallAUC"):
    metrics.PrecisionRecallAUC = metrics.RollingPRAUC

metric_sgd = metrics.PrecisionRecallAUC(window_size=1000, pos_val=1)
metric_adam = metrics.PrecisionRecallAUC(window_size=1000, pos_val=1)
metric_rf = metrics.PrecisionRecallAUC(window_size=1000, pos_val=1)

cm_sgd = metrics.ConfusionMatrix()
cm_adam = metrics.ConfusionMatrix()
cm_rf = metrics.ConfusionMatrix()
# Dynamic Engine Configuration
engine_config = {
    "block_threshold": 0.70,
    "alert_budget_alpha": 1.0,
    "window_size": 500,
}

class ConfigUpdate(BaseModel):
    block_threshold: Optional[float] = None
    alert_budget_alpha: Optional[float] = None
    window_size: Optional[int] = None

# Window buffer for Pr@k computation
labeled_buffer = []
BUFFER_MAX = 10000
PREFIX_FRAUD_RATE = 0.00189  # Fraud rate in 50% prefix dataset (269 / 142403)

def compute_pr_at_k(window: list, score_key: str = "prob_sgd", alpha: float = 1.0, fraud_rate: float = PREFIX_FRAUD_RATE) -> tuple[float, int]:
    """Calculate Precision at k (Pr@k) on current transaction window.
    k = max(1, round(alpha * prefix_fraud_rate * window_size))
    """
    if not window:
        return 0.0, 1
    m = len(window)
    k = max(1, int(round(alpha * fraud_rate * m)))
    sorted_window = sorted(window, key=lambda item: item[score_key], reverse=True)
    top_k = sorted_window[:k]
    tp_in_k = sum(1 for item in top_k if item["label"] == 1)
    return round(tp_in_k / k, 6), k

transaction_log = []
LOG_MAX = 500
log_lock = threading.Lock()

stats = {
    "total": 0,
    "fraud_detected": 0,
    "blocked": 0,
    "prauc_sgd": 0.0,
    "prauc_adam": 0.0,
    "prauc_rf": 0.0,
    "sk_ap_sgd": 0.0,
    "sk_ap_adam": 0.0,
    "sk_ap_rf": 0.0,
    "pr_at_k_sgd": 0.0,
    "pr_at_k_adam": 0.0,
    "pr_at_k_rf": 0.0,
    "k_eval": 1,
    "window_size": 0,
    "frauds_in_window": 0,
    "cm_sgd": {"tp":0, "fp":0, "tn":0, "fn":0},
    "cm_adam": {"tp":0, "fp":0, "tn":0, "fn":0},
    "cm_rf": {"tp":0, "fp":0, "tn":0, "fn":0},
    "config": engine_config,
    # Backward compatibility aliases
    "roc_auc_sgd": 0.0,
    "roc_auc_adam": 0.0,
    "acc_sgd": 0.0,
    "acc_adam": 0.0,
}

def recompute_window_metrics():
    """Recalculate dynamic window metrics immediately whenever window_size or alpha changes."""
    if not labeled_buffer:
        return
    win_len = engine_config.get("window_size", 500)
    alpha_val = engine_config.get("alert_budget_alpha", 1.0)
    active_window = labeled_buffer[-win_len:]
    
    pr_k_sgd, k_eval = compute_pr_at_k(active_window, "prob_sgd", alpha=alpha_val)
    pr_k_adam, _ = compute_pr_at_k(active_window, "prob_adam", alpha=alpha_val)
    
    y_win = [item["label"] for item in active_window]
    frauds_in_win = sum(y_win)
    if frauds_in_win > 0:
        sgd_scores = [item["prob_sgd"] for item in active_window]
        adam_scores = [item["prob_adam"] for item in active_window]
        stats["sk_ap_sgd"] = round(float(average_precision_score(y_win, sgd_scores)), 6)
        stats["sk_ap_adam"] = round(float(average_precision_score(y_win, adam_scores)), 6)
    else:
        stats["sk_ap_sgd"] = 0.0
        stats["sk_ap_adam"] = 0.0
        stats["sk_ap_rf"] = 0.0

    stats["pr_at_k_sgd"] = pr_k_sgd
    stats["pr_at_k_adam"] = pr_k_adam
    stats["pr_at_k_rf"] = pr_k_rf
    stats["k_eval"] = k_eval
    stats["window_size"] = len(active_window)
    stats["frauds_in_window"] = frauds_in_win
    stats["config"] = dict(engine_config)
    stats["acc_sgd"] = pr_k_sgd
    stats["acc_adam"] = pr_k_adam
    stats["acc_rf"] = pr_k_rf


class Transaction(BaseModel):
    features: dict  
    transaction_id: Optional[str] = None


class FeedbackPayload(BaseModel):
    features: dict
    label: int 
    transaction_id: Optional[str] = None



@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "engine": "River Online Machine Learning",
        "deployment_target": "Payment Edge Gateways / Cloud Message Brokers (Kafka/Flink)",
        "evaluation_standard": "Experimental Protocol v2 (PR-AUC, Pr@k)"
    }


@app.post("/predict", tags=["Inference"])
def predict_transaction(tx: Transaction):
    x = tx.features
    try:
        proba_sgd = model_sgd.predict_proba_one(x)
        prob_sgd = proba_sgd.get(1, 0.0) if proba_sgd else 0.0
        
        proba_adam = model_adam.predict_proba_one(x)
        prob_adam = proba_adam.get(1, 0.0) if proba_adam else 0.0
        
        # Lấy xác suất từ Random Forest
        prob_rf_array = model_rf.predict_proba([x])
        prob_rf = float(prob_rf_array[0][1])
    except Exception:
        prob_sgd = 0.0
        prob_adam = 0.0
        prob_rf = 0.0
        
    threshold = engine_config.get("block_threshold", 0.70)
    action = "BLOCK" if prob_sgd > threshold else "ALLOW"
    record = {
        "transaction_id": tx.transaction_id,
        "fraud_probability": round(prob_sgd, 6),
        "prob_sgd": round(prob_sgd, 6),
        "prob_adam": round(prob_adam, 6),
        "prob_rf": round(prob_rf, 6), # <-- Thêm RF
        "action": action,
        "timestamp": time.time(),
    }
    with log_lock:
        transaction_log.append(record)
        if len(transaction_log) > LOG_MAX:
            transaction_log.pop(0)
        stats["total"] += 1
        if action == "BLOCK":
            stats["blocked"] += 1
    return record


@app.post("/feedback", tags=["Learning"])
def feedback_label(payload: FeedbackPayload):
    x = payload.features
    y = payload.label
    try:
        proba_sgd = model_sgd.predict_proba_one(x)
        prob_sgd = proba_sgd.get(1, 0.0) if proba_sgd else 0.0
        model_sgd.learn_one(x, y)
        metric_sgd.update(y, prob_sgd)
        cm_sgd.update(y, 1 if prob_sgd > 0.5 else 0)
        
        proba_adam = model_adam.predict_proba_one(x)
        prob_adam = proba_adam.get(1, 0.0) if proba_adam else 0.0
        model_adam.learn_one(x, y)
        metric_adam.update(y, prob_adam)
        cm_adam.update(y, 1 if prob_adam > 0.5 else 0)
        
        prob_rf_array = model_rf.predict_proba([x])
        prob_rf = float(prob_rf_array[0][1])
        metric_rf.update(y, prob_rf)
        cm_rf.update(y, 1 if prob_rf > 0.5 else 0)
        
        with log_lock:
            labeled_buffer.append({
                "prob_sgd": prob_sgd,
                "prob_adam": prob_adam,
                "prob_rf": prob_rf, # <-- Thêm RF
                "label": y
            })
            if len(labeled_buffer) > BUFFER_MAX:
                labeled_buffer.pop(0)

            # Recalculate dynamic window metrics
            recompute_window_metrics()

            if y == 1:
                stats["fraud_detected"] += 1
            stats["prauc_sgd"] = round(metric_sgd.get(), 6)
            stats["prauc_adam"] = round(metric_adam.get(), 6)
            stats["prauc_rf"] = round(metric_rf.get(), 6)
            stats["roc_auc_sgd"] = stats["prauc_sgd"]
            stats["roc_auc_adam"] = stats["prauc_adam"]
            stats["roc_auc_rf"] = stats["prauc_rf"]
            
            # Get confusion matrix details (True/False Positives/Negatives)
            stats["cm_sgd"] = {
                "tp": cm_sgd.true_positives(1),
                "fp": cm_sgd.false_positives(1),
                "tn": cm_sgd.true_negatives(1),
                "fn": cm_sgd.false_negatives(1)
            }
            stats["cm_adam"] = {
                "tp": cm_adam.true_positives(1),
                "fp": cm_adam.false_positives(1),
                "tn": cm_adam.true_negatives(1),
                "fn": cm_adam.false_negatives(1)
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "model_updated",
        "prauc_sgd": stats["prauc_sgd"],
        "prauc_adam": stats["prauc_adam"],
        "prauc_rf": stats.get("prauc_rf", 0.0),
        "pr_at_k_sgd": stats["pr_at_k_sgd"],
        "pr_at_k_adam": stats["pr_at_k_adam"],
        "pr_at_k_rf": stats.get("pr_at_k_rf", 0.0),
        "k_eval": stats["k_eval"],
        "transaction_id": payload.transaction_id,
    }


@app.get("/stats", tags=["Monitoring"])
def get_stats():
    with log_lock:
        return {
            **stats,
            "log_size": len(transaction_log),
        }


@app.get("/logs", tags=["Monitoring"])
def get_logs(limit: int = 50):
    with log_lock:
        return {"transactions": transaction_log[-limit:]}


@app.get("/config", tags=["Configuration"])
def get_config():
    with log_lock:
        return dict(engine_config)


@app.post("/config", tags=["Configuration"])
def update_config(cfg: ConfigUpdate):
    with log_lock:
        if cfg.block_threshold is not None:
            engine_config["block_threshold"] = max(0.01, min(0.99, float(cfg.block_threshold)))
        if cfg.alert_budget_alpha is not None:
            engine_config["alert_budget_alpha"] = max(0.01, float(cfg.alert_budget_alpha))
        if cfg.window_size is not None:
            engine_config["window_size"] = max(50, int(cfg.window_size))
        
        # Tinh toan lai metrics ngay lap tuc cho window_size / alpha moi
        recompute_window_metrics()

        return {
            "status": "config_updated",
            "config": dict(engine_config),
            "stats": stats
        }


@app.get("/pr_curve", tags=["Monitoring"])
def get_pr_curve(limit: Optional[int] = None):
    with log_lock:
        win_size = limit if limit is not None else engine_config.get("window_size", 500)
        window_sample = labeled_buffer[-win_size:] if labeled_buffer else []
        if not window_sample:
            return {"sgd": {"precision": [], "recall": []}, "adam": {"precision": [], "recall": []}, "rf": {"precision": [], "recall": []}, "frauds_in_window": 0, "window_size": 0} 

        y_w = [item["label"] for item in window_sample]
        frauds_count = sum(y_w)
        if frauds_count == 0:
            return {"sgd": {"precision": [], "recall": []}, "adam": {"precision": [], "recall": []}, "rf": {"precision": [], "recall": []}, "frauds_in_window": 0, "window_size": len(window_sample)}
            
        sgd_s = [item["prob_sgd"] for item in window_sample]
        adam_s = [item["prob_adam"] for item in window_sample]
        rf_s = [item["prob_rf"] for item in window_sample]
        
        p_sgd, r_sgd, _ = precision_recall_curve(y_w, sgd_s)
        p_adam, r_adam, _ = precision_recall_curve(y_w, adam_s)
        p_rf, r_rf, _ = precision_recall_curve(y_w, rf_s)
        
        return {
            "sgd": {"precision": [round(float(v), 4) for v in p_sgd], "recall": [round(float(v), 4) for v in r_sgd]},
            "adam": {"precision": [round(float(v), 4) for v in p_adam], "recall": [round(float(v), 4) for v in r_adam]},
            "rf": {"precision": [round(float(v), 4) for v in p_rf], "recall": [round(float(v), 4) for v in r_rf]},
            "frauds_in_window": frauds_count,
            "window_size": len(window_sample)
        }
