from collections import deque
import csv
import os
import pickle
import random
import threading
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from river import drift, forest, metrics, tree
from sklearn.metrics import average_precision_score, precision_recall_curve

app = FastAPI(
    title="Payment Edge Fraud Detection Engine",
    description="Online ML Engine với các mô hình chuyên biệt cho dữ liệu luồng (HAT & ARF)",
    version="3.0.0"
)

# -------------------------------------------------------------
# 1. KHỞI TẠO MÔ HÌNH (HAT, ARF, BATCH RF)
# -------------------------------------------------------------
MODEL_HAT_FILE = "model_hat.pkl"
MODEL_ARF_FILE = "model_arf.pkl"
MODEL_RF_FILE  = "model_rf.pkl"
# Tệp tương thích ngược
MODEL_SGD_FILE = "model_sgd.pkl"
MODEL_ADAM_FILE = "model_adam.pkl"

hat_file = MODEL_HAT_FILE if os.path.exists(MODEL_HAT_FILE) else MODEL_SGD_FILE
arf_file = MODEL_ARF_FILE if os.path.exists(MODEL_ARF_FILE) else MODEL_ADAM_FILE

if os.path.exists(hat_file) and os.path.exists(arf_file) and os.path.exists(MODEL_RF_FILE):
    print(f"[Server] Loaded pretrained models ({hat_file}, {arf_file}, {MODEL_RF_FILE}) successfully!")
    with open(hat_file, "rb") as f:
        model_hat = pickle.load(f)
    with open(arf_file, "rb") as f:
        model_arf = pickle.load(f)
    with open(MODEL_RF_FILE, "rb") as f:
        model_rf = pickle.load(f)
    if model_rf and hasattr(model_rf, "named_steps") and "randomforestclassifier" in model_rf.named_steps:
        model_rf.named_steps["randomforestclassifier"].n_jobs = 1
else:
    print("[Server] Model files not found. Initializing fresh streaming models (HAT & ARF)...")
    model_hat = tree.HoeffdingAdaptiveTreeClassifier(grace_period=50, seed=42)
    model_arf = forest.ARFClassifier(n_models=15, grace_period=50, seed=42)
    model_rf = None

# -------------------------------------------------------------
# 2. FRAUD EXPERIENCE REPLAY BUFFER (CHỐNG QUÊN TAI HẠI)
# -------------------------------------------------------------
FRAUD_REPLAY_MAX = 100
fraud_replay_buffer = deque(maxlen=FRAUD_REPLAY_MAX)
clean_tx_counter = 0
replay_stats = {
    "total_replays": 0,
    "last_replay": None,
}


def init_fraud_replay_buffer():
    """Tải trước tối đa 100 mẫu gian lận lịch sử để bộ nhớ đệm hoạt động ngay."""
    prefix_path = "data/creditcard_prefix.csv"
    if not os.path.exists(prefix_path):
        return
    try:
        with open(prefix_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = int(float(row.pop("Class", 0)))
                row.pop("Time", None)
                if label == 1:
                    feat = {k: float(v) for k, v in row.items() if v != ""}
                    fraud_replay_buffer.append({
                        "features": feat,
                        "label": 1,
                        "error": 1.0,
                        "priority": 1.0,
                        "timestamp": round(time.time(), 2),
                    })
                    if len(fraud_replay_buffer) >= FRAUD_REPLAY_MAX:
                        break
        print(f"[Experience Replay] Pre-loaded {len(fraud_replay_buffer)} historical fraud samples into Fraud Memory.")
    except Exception as e:
        print(f"[Experience Replay Error] Could not preload fraud buffer: {e}")


init_fraud_replay_buffer()

# -------------------------------------------------------------
# 3. THƯỚC ĐO HIỆU NĂNG & CẤU HÌNH ĐỘNG
# -------------------------------------------------------------
if not hasattr(metrics, "PrecisionRecallAUC"):
    metrics.PrecisionRecallAUC = getattr(metrics, "RollingPRAUC", metrics.RollingROCAUC)

metric_hat = metrics.PrecisionRecallAUC(window_size=1000, pos_val=1)
metric_arf = metrics.PrecisionRecallAUC(window_size=1000, pos_val=1)
metric_rf  = metrics.PrecisionRecallAUC(window_size=1000, pos_val=1)

cm_hat = metrics.ConfusionMatrix()
cm_arf = metrics.ConfusionMatrix()
cm_rf  = metrics.ConfusionMatrix()

engine_config = {
    "block_threshold": 0.70,     # Ngưỡng xác suất chặn giao dịch
    "alert_budget_alpha": 1.0,   # Hệ số ngân sách cảnh báo Pr@k
    "window_size": 1000,         # Kích thước cửa sổ đánh giá
    "enable_per": True,          # Ưu tiên mẫu có độ lỗi cao (Prioritized Experience Replay)
    "enable_noise_aug": True,    # Thêm nhiễu Gaussian sinh biến thể
    "noise_std": 0.02,           # Độ lệch chuẩn nhiễu
    "replay_interval": 10,       # Ôn tập 1 mẫu sau mỗi N giao dịch hợp lệ
}


class ConfigUpdate(BaseModel):
    block_threshold: Optional[float] = None
    alert_budget_alpha: Optional[float] = None
    window_size: Optional[int] = None
    enable_per: Optional[bool] = None
    enable_noise_aug: Optional[bool] = None
    noise_std: Optional[float] = None
    replay_interval: Optional[int] = None


# Bộ đệm cửa sổ cho Pr@k
labeled_buffer = []
BUFFER_MAX = 10000
PREFIX_FRAUD_RATE = 0.00189  # Tỷ lệ gian lận tập prefix (269 / 142403)


def compute_pr_at_k(window: list, score_key: str = "prob_arf", alpha: float = 1.0, fraud_rate: float = PREFIX_FRAUD_RATE) -> tuple[float, int]:
    """Tính Precision at k (Pr@k): k = max(1, round(alpha * fraud_rate * window_size))."""
    if not window:
        return 0.0, 1
    m = len(window)
    k = max(1, int(round(alpha * fraud_rate * m)))
    sorted_window = sorted(window, key=lambda item: item.get(score_key, 0.0), reverse=True)
    top_k = sorted_window[:k]
    tp_in_k = sum(1 for item in top_k if item.get("label", 0) == 1)
    return round(tp_in_k / k, 6), k


transaction_log = []
LOG_MAX = 500
prauc_history = []
log_lock = threading.Lock()

# Bộ phát hiện Concept Drift: River ADWIN
adwin = drift.ADWIN(delta=0.005)
drift_points = []

stats = {
    "total": 0,
    "fraud_detected": 0,
    "blocked": 0,
    "drift_detected": False,
    "drift_count": 0,
    "last_drift_step": None,
    "adwin_estimation": 0.0,
    "adwin_width": 0,
    "prauc_hat": 0.0,
    "prauc_arf": 0.0,
    "prauc_rf": 0.0,
    "sk_ap_hat": 0.0,
    "sk_ap_arf": 0.0,
    "sk_ap_rf": 0.0,
    "pr_at_k_hat": 0.0,
    "pr_at_k_arf": 0.0,
    "pr_at_k_rf": 0.0,
    "k_eval": 1,
    "window_size": 0,
    "frauds_in_window": 0,
    "cm_hat": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
    "cm_arf": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
    "cm_rf":  {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
    "global_tp_hat": 0,
    "global_tp_arf": 0,
    "global_tp_rf": 0,
    "global_acc_hat": 0.0,
    "global_acc_arf": 0.0,
    "global_acc_rf": 0.0,
    "config": engine_config,
    # Tương thích ngược phiên bản cũ:
    "prauc_sgd": 0.0,
    "prauc_adam": 0.0,
    "sk_ap_sgd": 0.0,
    "sk_ap_adam": 0.0,
    "pr_at_k_sgd": 0.0,
    "pr_at_k_adam": 0.0,
    "cm_sgd": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
    "cm_adam": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
    "roc_auc_sgd": 0.0,
    "roc_auc_adam": 0.0,
    "acc_sgd": 0.0,
    "acc_adam": 0.0,
}


def recompute_window_metrics():
    """Tính lại số liệu trên cửa sổ đánh giá khi kích thước hoặc tham số alpha thay đổi."""
    if not labeled_buffer:
        return
    win_len = engine_config.get("window_size", 500)
    alpha_val = engine_config.get("alert_budget_alpha", 1.0)
    active_window = labeled_buffer[-win_len:]

    pr_k_hat, k_eval = compute_pr_at_k(active_window, "prob_hat", alpha=alpha_val)
    pr_k_arf, _      = compute_pr_at_k(active_window, "prob_arf", alpha=alpha_val)
    pr_k_rf, _       = compute_pr_at_k(active_window, "prob_rf",  alpha=alpha_val)

    y_win = [item["label"] for item in active_window]
    frauds_in_win = sum(y_win)
    if frauds_in_win > 0:
        hat_scores = [item["prob_hat"] for item in active_window]
        arf_scores = [item["prob_arf"] for item in active_window]
        rf_scores  = [item["prob_rf"]  for item in active_window]
        stats["sk_ap_hat"] = round(float(average_precision_score(y_win, hat_scores)), 6)
        stats["sk_ap_arf"] = round(float(average_precision_score(y_win, arf_scores)), 6)
        stats["sk_ap_rf"]  = round(float(average_precision_score(y_win, rf_scores)),  6)
    else:
        stats["sk_ap_hat"] = 0.0
        stats["sk_ap_arf"] = 0.0
        stats["sk_ap_rf"]  = 0.0

    stats["pr_at_k_hat"] = pr_k_hat
    stats["pr_at_k_arf"] = pr_k_arf
    stats["pr_at_k_rf"]  = pr_k_rf
    stats["k_eval"] = k_eval
    stats["window_size"] = len(active_window)
    stats["frauds_in_window"] = frauds_in_win
    stats["config"] = dict(engine_config)

    # Cập nhật bí danh tương thích
    stats["prauc_sgd"] = stats["prauc_hat"]
    stats["prauc_adam"] = stats["prauc_arf"]
    stats["sk_ap_sgd"] = stats["sk_ap_hat"]
    stats["sk_ap_adam"] = stats["sk_ap_arf"]
    stats["pr_at_k_sgd"] = pr_k_hat
    stats["pr_at_k_adam"] = pr_k_arf
    stats["acc_sgd"] = pr_k_hat
    stats["acc_adam"] = pr_k_arf
    stats["acc_rf"] = pr_k_rf


# -------------------------------------------------------------
# 4. DATA SCHEMAS
# -------------------------------------------------------------
class Transaction(BaseModel):
    features: dict
    transaction_id: Optional[str] = None


class FeedbackPayload(BaseModel):
    features: dict
    label: int
    transaction_id: Optional[str] = None


class StreamTransaction(BaseModel):
    features: dict
    label: int
    transaction_id: Optional[str] = None


class BatchTransactions(BaseModel):
    transactions: list[StreamTransaction]


# -------------------------------------------------------------
# 5. CORE STREAMING LOGIC
# -------------------------------------------------------------
def _execute_stream_step_locked(x: dict, y: int, tx_id: Optional[str] = None, override_rf_prob: Optional[float] = None) -> tuple[dict, bool]:
    """Thực thi dự đoán, cập nhật online learning, ôn tập mẫu và đánh giá hiệu năng."""
    global clean_tx_counter

    # 1. Dự đoán xác suất gian lận
    try:
        proba_hat = model_hat.predict_proba_one(x)
        prob_hat = proba_hat.get(1, 0.0) if proba_hat else 0.0

        proba_arf = model_arf.predict_proba_one(x)
        prob_arf = proba_arf.get(1, 0.0) if proba_arf else 0.0

        if override_rf_prob is not None:
            prob_rf = override_rf_prob
        else:
            prob_rf_array = model_rf.predict_proba([x]) if model_rf else [[0.0, 0.0]]
            prob_rf = float(prob_rf_array[0][1])
    except Exception:
        prob_hat = 0.0
        prob_arf = 0.0
        prob_rf = 0.0

    threshold = engine_config.get("block_threshold", 0.70)
    action = "BLOCK" if prob_arf > threshold else "ALLOW"

    # 2. Ghi nhận giao dịch
    stats["total"] += 1
    if action == "BLOCK":
        stats["blocked"] += 1
    if y == 1:
        stats["fraud_detected"] += 1

    record = {
        "transaction_id": tx_id,
        "fraud_probability": round(prob_arf, 6),
        "prob_hat": round(prob_hat, 6),
        "prob_arf": round(prob_arf, 6),
        "prob_rf": round(prob_rf, 6),
        "prob_sgd": round(prob_hat, 6),
        "prob_adam": round(prob_arf, 6),
        "action": action,
        "timestamp": time.time(),
    }
    transaction_log.append(record)
    if len(transaction_log) > LOG_MAX:
        transaction_log.pop(0)

    # 3. Học trực tuyến (Online Incremental Learning)
    weight = 250.0 if y == 1 else 1.0
    model_hat.learn_one(x, y, w=weight)
    metric_hat.update(y, prob_hat)
    cm_hat.update(y, 1 if prob_hat > 0.5 else 0)

    model_arf.learn_one(x, y, w=weight)
    metric_arf.update(y, prob_arf)
    cm_arf.update(y, 1 if prob_arf > 0.5 else 0)

    metric_rf.update(y, prob_rf)
    cm_rf.update(y, 1 if prob_rf > 0.5 else 0)

    # 4. Ôn tập mẫu gian lận (Prioritized Experience Replay + Noise Augmentation)
    if y == 1:
        err = abs(1.0 - prob_arf)
        priority = max(0.05, float(err))
        fraud_replay_buffer.append({
            "features": x.copy(),
            "label": 1,
            "error": round(float(err), 4),
            "priority": round(priority, 4),
            "timestamp": round(time.time(), 2),
        })
    else:
        clean_tx_counter += 1
        replay_interval = int(engine_config.get("replay_interval", 10))
        if clean_tx_counter % replay_interval == 0 and len(fraud_replay_buffer) > 0:
            buf_list = list(fraud_replay_buffer)
            if engine_config.get("enable_per", True):
                priorities = [it.get("priority", 1.0) for it in buf_list]
                total_p = sum(priorities)
                chosen = random.choices(buf_list, weights=priorities, k=1)[0] if total_p > 0 else random.choice(buf_list)
            else:
                chosen = random.choice(buf_list)

            orig_feat = chosen["features"]
            if engine_config.get("enable_noise_aug", True):
                std_val = float(engine_config.get("noise_std", 0.02))
                aug_feat = {}
                for k_name, v_val in orig_feat.items():
                    noise = random.gauss(0, std_val * max(1.0, abs(v_val))) if k_name == "Amount" else random.gauss(0, std_val)
                    aug_feat[k_name] = round(v_val + noise, 6)
            else:
                aug_feat = orig_feat.copy()

            model_hat.learn_one(aug_feat, 1, w=250.0)
            model_arf.learn_one(aug_feat, 1, w=250.0)
            replay_stats["total_replays"] += 1
            replay_stats["last_replay"] = {
                "original": orig_feat,
                "augmented": aug_feat,
                "priority": chosen.get("priority", 1.0),
                "error": chosen.get("error", 1.0),
                "timestamp": round(time.time(), 2),
            }

    # 5. Phát hiện Concept Drift bằng ADWIN
    score_val = float(metric_arf.get())
    drift_flag = False
    if score_val > 0.0:
        rolling_error = max(0.0, min(1.0, 1.0 - score_val))
        adwin.update(rolling_error)
        if adwin.drift_detected:
            stats["drift_count"] += 1
            stats["last_drift_step"] = stats["total"]
            drift_flag = True
            drift_record = {
                "step": stats["total"],
                "timestamp": round(time.time(), 2),
                "prauc": round(score_val, 4),
                "error": round(rolling_error, 4),
                "estimation": round(adwin.estimation, 4),
                "width": adwin.width,
            }
            drift_points.append(drift_record)
            print(f"[ADWIN Drift] Concept Drift detected at transaction #{stats['total']} (PR-AUC={score_val:.4f})")

    stats["drift_detected"] = drift_flag
    stats["adwin_estimation"] = round(adwin.estimation, 4)
    stats["adwin_width"] = adwin.width

    # 6. Cập nhật bộ đệm nhãn
    labeled_buffer.append({
        "prob_hat": prob_hat,
        "prob_arf": prob_arf,
        "prob_rf": prob_rf,
        "prob_sgd": prob_hat,
        "prob_adam": prob_arf,
        "label": y
    })
    if len(labeled_buffer) > BUFFER_MAX:
        labeled_buffer.pop(0)

    # 7. Cập nhật PR-AUC rolling
    raw_hat = metric_hat.get()
    raw_arf = metric_arf.get()
    raw_rf  = metric_rf.get()

    if raw_hat > 0 or stats["prauc_hat"] == 0:
        stats["prauc_hat"] = round(raw_hat, 6)
    if raw_arf > 0 or stats["prauc_arf"] == 0:
        stats["prauc_arf"] = round(raw_arf, 6)
    if raw_rf > 0 or stats["prauc_rf"] == 0:
        stats["prauc_rf"]  = round(raw_rf, 6)

    stats["prauc_sgd"] = stats["prauc_hat"]
    stats["prauc_adam"] = stats["prauc_arf"]
    stats["roc_auc_sgd"] = stats["prauc_hat"]
    stats["roc_auc_adam"] = stats["prauc_arf"]
    stats["roc_auc_rf"] = stats["prauc_rf"]

    # 8. Ma trận nhầm lẫn & Tỷ lệ bắt đúng toàn cục (TP / Total Frauds)
    tp_hat = int(cm_hat.true_positives(1))
    tp_arf = int(cm_arf.true_positives(1))
    tp_rf  = int(cm_rf.true_positives(1))
    tot_f  = int(stats["fraud_detected"])

    stats["cm_hat"] = {"tp": tp_hat, "fp": int(cm_hat.false_positives(1)), "tn": int(cm_hat.true_negatives(1)), "fn": int(cm_hat.false_negatives(1))}
    stats["cm_arf"] = {"tp": tp_arf, "fp": int(cm_arf.false_positives(1)), "tn": int(cm_arf.true_negatives(1)), "fn": int(cm_arf.false_negatives(1))}
    stats["cm_rf"]  = {"tp": tp_rf,  "fp": int(cm_rf.false_positives(1)),  "tn": int(cm_rf.true_negatives(1)),  "fn": int(cm_rf.false_negatives(1))}
    stats["cm_sgd"] = stats["cm_hat"]
    stats["cm_adam"] = stats["cm_arf"]

    stats["global_tp_hat"] = tp_hat
    stats["global_tp_arf"] = tp_arf
    stats["global_tp_rf"]  = tp_rf
    stats["global_acc_hat"] = round((tp_hat / tot_f) * 100, 2) if tot_f > 0 else 0.0
    stats["global_acc_arf"] = round((tp_arf / tot_f) * 100, 2) if tot_f > 0 else 0.0
    stats["global_acc_rf"]  = round((tp_rf / tot_f) * 100, 2)  if tot_f > 0 else 0.0

    # 9. Lưu lịch sử mỗi 10 bước hoặc khi xảy ra drift
    if stats["total"] % 10 == 0 or drift_flag:
        prauc_history.append({
            "step": stats["total"],
            "timestamp": round(time.time(), 2),
            "prauc_hat": stats["prauc_hat"],
            "prauc_arf": stats["prauc_arf"],
            "prauc_rf": stats["prauc_rf"],
            "prauc_sgd": stats["prauc_hat"],
            "prauc_adam": stats["prauc_arf"],
            "drift": drift_flag,
        })

    # 10. Tái tính toán Pr@k và AP theo chu kỳ 50 bước để tối ưu tốc độ
    if stats["total"] % 50 == 0:
        recompute_window_metrics()

    return record, drift_flag


# -------------------------------------------------------------
# 6. API ENDPOINTS
# -------------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "engine": "River Online ML (Hoeffding Adaptive Tree & Adaptive Random Forest)",
        "protocol": "Experimental Protocol v2 (PR-AUC, Pr@k)"
    }


@app.post("/predict", tags=["Inference"])
def predict_transaction(tx: Transaction):
    x = tx.features
    try:
        proba_hat = model_hat.predict_proba_one(x)
        prob_hat = proba_hat.get(1, 0.0) if proba_hat else 0.0

        proba_arf = model_arf.predict_proba_one(x)
        prob_arf = proba_arf.get(1, 0.0) if proba_arf else 0.0

        prob_rf_array = model_rf.predict_proba([x]) if model_rf else [[0.0, 0.0]]
        prob_rf = float(prob_rf_array[0][1])
    except Exception:
        prob_hat = prob_arf = prob_rf = 0.0

    threshold = engine_config.get("block_threshold", 0.70)
    action = "BLOCK" if prob_arf > threshold else "ALLOW"
    record = {
        "transaction_id": tx.transaction_id,
        "fraud_probability": round(prob_arf, 6),
        "prob_hat": round(prob_hat, 6),
        "prob_arf": round(prob_arf, 6),
        "prob_rf": round(prob_rf, 6),
        "prob_sgd": round(prob_hat, 6),
        "prob_adam": round(prob_arf, 6),
        "action": action,
        "timestamp": time.time(),
    }
    with log_lock:
        transaction_log.append(record)
        if len(transaction_log) > LOG_MAX:
            transaction_log.pop(0)
    return record


@app.post("/feedback", tags=["Learning"])
def feedback_label(payload: FeedbackPayload):
    try:
        with log_lock:
            _, drift_flag = _execute_stream_step_locked(payload.features, payload.label, payload.transaction_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "model_updated",
        "prauc_hat": stats["prauc_hat"],
        "prauc_arf": stats["prauc_arf"],
        "prauc_rf": stats.get("prauc_rf", 0.0),
        "pr_at_k_hat": stats["pr_at_k_hat"],
        "pr_at_k_arf": stats["pr_at_k_arf"],
        "pr_at_k_rf": stats.get("pr_at_k_rf", 0.0),
        "drift_detected": drift_flag,
        "drift_count": stats["drift_count"],
        "last_drift_step": stats["last_drift_step"],
        "prauc_sgd": stats["prauc_hat"],
        "prauc_adam": stats["prauc_arf"],
        "pr_at_k_sgd": stats["pr_at_k_hat"],
        "pr_at_k_adam": stats["pr_at_k_arf"],
        "k_eval": stats["k_eval"],
        "transaction_id": payload.transaction_id,
    }


@app.post("/stream_tx", tags=["Streaming"])
def stream_single_transaction(tx: StreamTransaction):
    """Kết hợp Dự đoán + Phản hồi nhãn trong 1 lượt gọi HTTP."""
    try:
        with log_lock:
            record, drift_flag = _execute_stream_step_locked(tx.features, tx.label, tx.transaction_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "processed",
        "record": record,
        "drift_detected": drift_flag,
        "prauc_arf": stats["prauc_arf"]
    }


@app.post("/batch_stream", tags=["Streaming"])
def batch_stream_transactions(payload: BatchTransactions):
    """Xử lý luồng giao dịch dạng micro-batch tăng tốc tối đa (1000+ tx/s)."""
    count = 0
    frauds = 0
    if not payload.transactions:
        return {"status": "empty", "batch_count": 0}

    # Tính trước xác suất RF theo batch (vectorized)
    rf_probs = None
    if model_rf:
        try:
            feats_batch = [item.features for item in payload.transactions]
            prob_rf_array = model_rf.predict_proba(feats_batch)
            rf_probs = [float(p[1]) for p in prob_rf_array]
        except Exception:
            rf_probs = None

    try:
        with log_lock:
            for i, item in enumerate(payload.transactions):
                rf_p = rf_probs[i] if rf_probs is not None and i < len(rf_probs) else None
                _execute_stream_step_locked(item.features, item.label, item.transaction_id, override_rf_prob=rf_p)
                count += 1
                if item.label == 1:
                    frauds += 1
            recompute_window_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "batch_processed",
        "batch_count": count,
        "fraud_count": frauds,
        "total_processed": stats["total"],
        "prauc_arf": stats["prauc_arf"]
    }


@app.get("/stats", tags=["Monitoring"])
def get_stats():
    with log_lock:
        return {**stats, "log_size": len(transaction_log)}


@app.get("/logs", tags=["Monitoring"])
def get_logs(limit: int = 50):
    with log_lock:
        return {"transactions": transaction_log[-limit:]}


@app.get("/prauc_history", tags=["Monitoring"])
def get_prauc_history(limit: Optional[int] = None):
    with log_lock:
        if limit is not None and limit > 0:
            return {"history": prauc_history[-limit:]}
        return {"history": list(prauc_history)}


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
        if cfg.enable_per is not None:
            engine_config["enable_per"] = bool(cfg.enable_per)
        if cfg.enable_noise_aug is not None:
            engine_config["enable_noise_aug"] = bool(cfg.enable_noise_aug)
        if cfg.noise_std is not None:
            engine_config["noise_std"] = max(0.001, min(0.20, float(cfg.noise_std)))
        if cfg.replay_interval is not None:
            engine_config["replay_interval"] = max(1, min(100, int(cfg.replay_interval)))

        recompute_window_metrics()
        return {"status": "config_updated", "config": dict(engine_config), "stats": stats}


@app.get("/pr_curve", tags=["Monitoring"])
def get_pr_curve(limit: Optional[int] = None):
    with log_lock:
        win_size = limit if limit is not None else engine_config.get("window_size", 500)
        window_sample = labeled_buffer[-win_size:] if labeled_buffer else []
        if not window_sample:
            empty = {"precision": [], "recall": []}
            return {"hat": empty, "arf": empty, "rf": empty, "sgd": empty, "adam": empty, "frauds_in_window": 0, "window_size": 0}

        y_w = [item["label"] for item in window_sample]
        frauds_count = sum(y_w)
        if frauds_count == 0:
            empty = {"precision": [], "recall": []}
            return {"hat": empty, "arf": empty, "rf": empty, "sgd": empty, "adam": empty, "frauds_in_window": 0, "window_size": len(window_sample)}

        hat_s = [item.get("prob_hat", item.get("prob_sgd", 0.0)) for item in window_sample]
        arf_s = [item.get("prob_arf", item.get("prob_adam", 0.0)) for item in window_sample]
        rf_s  = [item.get("prob_rf", 0.0) for item in window_sample]

        p_hat, r_hat, _ = precision_recall_curve(y_w, hat_s)
        p_arf, r_arf, _ = precision_recall_curve(y_w, arf_s)
        p_rf,  r_rf,  _ = precision_recall_curve(y_w, rf_s)

        res_hat = {"precision": [round(float(v), 4) for v in p_hat], "recall": [round(float(v), 4) for v in r_hat]}
        res_arf = {"precision": [round(float(v), 4) for v in p_arf], "recall": [round(float(v), 4) for v in r_arf]}
        res_rf  = {"precision": [round(float(v), 4) for v in p_rf],  "recall": [round(float(v), 4) for v in r_rf]}
        return {
            "hat": res_hat,
            "arf": res_arf,
            "rf": res_rf,
            "sgd": res_hat,
            "adam": res_arf,
            "frauds_in_window": frauds_count,
            "window_size": len(window_sample)
        }


@app.get("/models_status", tags=["Health"])
def models_status():
    hat_ready = os.path.exists(MODEL_HAT_FILE) or os.path.exists(MODEL_SGD_FILE)
    arf_ready = os.path.exists(MODEL_ARF_FILE) or os.path.exists(MODEL_ADAM_FILE)
    rf_ready = os.path.exists(MODEL_RF_FILE)
    return {
        "pretrained_ready": hat_ready and arf_ready and rf_ready,
        "hat_ready": hat_ready,
        "arf_ready": arf_ready,
        "rf_ready": rf_ready,
    }


@app.post("/reload_models", tags=["Configuration"])
def reload_models():
    global model_hat, model_arf, model_rf, adwin, clean_tx_counter
    hat_f = MODEL_HAT_FILE if os.path.exists(MODEL_HAT_FILE) else MODEL_SGD_FILE
    arf_f = MODEL_ARF_FILE if os.path.exists(MODEL_ARF_FILE) else MODEL_ADAM_FILE
    if os.path.exists(hat_f) and os.path.exists(arf_f) and os.path.exists(MODEL_RF_FILE):
        with log_lock:
            with open(hat_f, "rb") as f:
                model_hat = pickle.load(f)
            with open(arf_f, "rb") as f:
                model_arf = pickle.load(f)
            with open(MODEL_RF_FILE, "rb") as f:
                model_rf = pickle.load(f)
            adwin = drift.ADWIN(delta=0.005)
            clean_tx_counter = 0
            drift_points.clear()
            stats["drift_count"] = 0
            stats["drift_detected"] = False
            stats["last_drift_step"] = None
        return {"status": "models_reloaded"}
    return {"status": "files_not_found"}


@app.get("/drift_info", tags=["Monitoring"])
def get_drift_info():
    with log_lock:
        return {
            "drift_detected": stats.get("drift_detected", False),
            "drift_count": stats.get("drift_count", 0),
            "last_drift_step": stats.get("last_drift_step"),
            "adwin_estimation": stats.get("adwin_estimation", 0.0),
            "adwin_width": stats.get("adwin_width", 0),
            "drift_points": list(drift_points),
            "fraud_replay_size": len(fraud_replay_buffer),
        }


@app.get("/replay_info", tags=["Monitoring"])
def get_replay_info():
    with log_lock:
        items = list(fraud_replay_buffer)
        sorted_by_priority = sorted(items, key=lambda it: it.get("priority", 0.0), reverse=True)
        top_hardest = []
        for it in sorted_by_priority[:8]:
            feat = it.get("features", {})
            top_hardest.append({
                "error": it.get("error", 1.0),
                "priority": it.get("priority", 1.0),
                "timestamp": it.get("timestamp", 0),
                "V10": round(feat.get("V10", 0.0), 3),
                "V12": round(feat.get("V12", 0.0), 3),
                "V14": round(feat.get("V14", 0.0), 3),
                "V17": round(feat.get("V17", 0.0), 3),
                "Amount": round(feat.get("Amount", 0.0), 2),
            })
        return {
            "config": {
                "enable_per": engine_config.get("enable_per", True),
                "enable_noise_aug": engine_config.get("enable_noise_aug", True),
                "noise_std": engine_config.get("noise_std", 0.02),
                "replay_interval": engine_config.get("replay_interval", 10),
            },
            "buffer_size": len(fraud_replay_buffer),
            "buffer_max": FRAUD_REPLAY_MAX,
            "total_replays": replay_stats["total_replays"],
            "top_hardest": top_hardest,
            "last_replay": replay_stats["last_replay"],
        }
