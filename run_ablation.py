import argparse
import copy
import csv
import os
import pickle
import random
import time
from collections import deque
from queue import PriorityQueue

import numpy as np
from river import drift, metrics
from sklearn.metrics import average_precision_score

# Tuong thich phien ban River
if not hasattr(metrics, "PrecisionRecallAUC"):
    metrics.PrecisionRecallAUC = getattr(metrics, "RollingPRAUC", metrics.RollingROCAUC)

PREFIX_FILE = "data/creditcard_prefix.csv"
SUFFIX_FILE = "data/creditcard_suffix.csv"
MODEL_ARF_FILE = "model_arf.pkl"
MODEL_RF_FILE = "model_rf.pkl"
PREFIX_FRAUD_RATE = 0.00189  # 269 / 142403


def parse_row(row: dict) -> tuple[dict, int]:
    label = int(float(row.pop("Class", 0)))
    row.pop("Time", None)
    features = {k: float(v) for k, v in row.items() if v != ""}
    return features, label


def load_historical_frauds(prefix_path: str, max_items: int = 100) -> list[dict]:
    items = []
    if not os.path.exists(prefix_path):
        return items
    with open(prefix_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feat, label = parse_row(row)
            if label == 1:
                items.append({
                    "features": feat,
                    "label": 1,
                    "error": 1.0,
                    "priority": 1.0,
                })
                if len(items) >= max_items:
                    break
    return items


def build_shared_stream(
    data_path: str,
    rows: int,
    drift_enabled: bool,
    drift_start: int,
    drift_len: int,
    seed: int,
) -> list[dict]:
    """Tao duy nhat 1 luong du lieu + lich tre nhan co dinh de M0-M6 chay cong bang."""
    np.random.seed(seed)
    stream = []
    with open(data_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, raw_row in enumerate(reader):
            if idx >= rows:
                break
            features, true_label = parse_row(raw_row)

            # Giong het logic Concept Drift cua payment_edge.py / boost.py
            if drift_enabled and drift_start <= idx < (drift_start + drift_len):
                if true_label == 1:
                    for k in ["V10", "V12", "V14", "V17"]:
                        if k in features:
                            features[k] = -features[k] * 1.5
                elif idx % 10 == 0:
                    true_label = 1
                    for k in ["V4", "V11"]:
                        if k in features:
                            features[k] = -features[k]

            step = idx + 1
            if true_label == 0:
                delay_steps = int(max(1, round(np.random.normal(50, 10))))
            else:
                delay_steps = int(max(1, round(np.random.lognormal(6.0, 0.5))))

            stream.append({
                "step": step,
                "features": features,
                "label": true_label,
                "due_step": step + delay_steps,
            })
    return stream


def compute_pr_at_k(window: list, alpha: float = 1.0, fraud_rate: float = PREFIX_FRAUD_RATE) -> tuple[float, int]:
    if not window:
        return 0.0, 1
    m = len(window)
    k = max(1, int(round(alpha * fraud_rate * m)))
    sorted_win = sorted(window, key=lambda item: item["prob"], reverse=True)
    tp_in_k = sum(1 for item in sorted_win[:k] if item["label"] == 1)
    return tp_in_k / k, k


def compute_tradeoff(tp: int, fp: int, tn: int, fn: int) -> dict:
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    f2 = (5 * precision * recall / (4 * precision + recall)) if (4 * precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "fpr": round(fpr, 6),
        "f1": round(f1, 6),
        "f2": round(f2, 6),
    }


def set_arf_grace_period(model_arf, gp: int):
    target = list(model_arf.steps.values())[-1] if hasattr(model_arf, "steps") and model_arf.steps else model_arf
    if hasattr(target, "grace_period"):
        target.grace_period = gp
    for member in getattr(target, "models", getattr(target, "data", [])):
        if hasattr(member, "grace_period"):
            member.grace_period = gp
        elif hasattr(member, "model") and hasattr(member.model, "grace_period"):
            member.model.grace_period = gp


def arf_learn_one_weighted(model_arf, x: dict, y: int, w: float = 1.0):
    """Cap nhat ARF co trong so thuc (khac phuc viec River ARFClassifier.learn_one bo qua **kwargs)."""
    if w == 1.0:
        model_arf.learn_one(x, y)
        return

    if len(model_arf) == 0:
        model_arf._init_ensemble(sorted(x.keys()))

    for i, tree_model in enumerate(model_arf):
        y_pred = tree_model.predict_one(x)
        model_arf._metrics[i].update(
            y_true=y,
            y_pred=(
                tree_model.predict_proba_one(x)
                if isinstance(model_arf.metric, metrics.base.ClassificationMetric)
                and not model_arf.metric.requires_labels
                else y_pred
            ),
        )

        k = int(np.random.poisson(model_arf.lambda_value))
        if k > 0:
            eff_w = float(k) * float(w)
            if not model_arf._warning_detection_disabled and model_arf._background[i] is not None:
                model_arf._background[i].learn_one(x=x, y=y, w=eff_w)
            tree_model.learn_one(x=x, y=y, w=eff_w)

            drift_input = None
            if not model_arf._warning_detection_disabled:
                drift_input = model_arf._drift_detector_input(i, y, y_pred)
                model_arf._warning_detectors[i].update(drift_input)
                if model_arf._warning_detectors[i].drift_detected:
                    model_arf._background[i] = model_arf._new_base_model()
                    model_arf._warning_detectors[i] = model_arf.warning_detector.clone()
                    model_arf._warning_tracker[i] += 1

            if not model_arf._drift_detection_disabled:
                drift_input = (
                    drift_input
                    if drift_input is not None
                    else model_arf._drift_detector_input(i, y, y_pred)
                )
                model_arf._drift_detectors[i].update(drift_input)
                if model_arf._drift_detectors[i].drift_detected:
                    if not model_arf._warning_detection_disabled and model_arf._background[i] is not None:
                        model_arf.data[i] = model_arf._background[i]
                        model_arf._background[i] = None
                        model_arf._warning_detectors[i] = model_arf.warning_detector.clone()
                        model_arf._drift_detectors[i] = model_arf.drift_detector.clone()
                        model_arf._metrics[i] = model_arf.metric.clone()
                    else:
                        model_arf.data[i] = model_arf._new_base_model()
                        model_arf._drift_detectors[i] = model_arf.drift_detector.clone()
                        model_arf._metrics[i] = model_arf.metric.clone()
                    model_arf._drift_tracker[i] += 1


def run_batch_rf(model_rf, stream: list[dict], window_size: int = 1000, alpha: float = 1.0, block_threshold: float = 0.70):
    """Chay baseline tinh Batch RF tren dung lich nhan tre cua luong."""
    print("\n[Ablation] Dang chay M0: Batch Random Forest (Baseline tinh)...")
    t0 = time.perf_counter()

    # Do do tre suy luan don le tren 200 mau dau cho cong bang voi OML
    sample_n = min(200, len(stream))
    lat_samples = []
    for i in range(sample_n):
        ts = time.perf_counter()
        _ = model_rf.predict_proba([stream[i]["features"]])[0][1]
        lat_samples.append((time.perf_counter() - ts) * 1000.0)
    avg_latency_ms = sum(lat_samples) / len(lat_samples) if lat_samples else 0.0

    # Du doan toan bo theo batch (ket qua giong het don le vi RF tinh)
    all_probs_rf = model_rf.predict_proba([item["features"] for item in stream])[:, 1]

    metric_prauc = metrics.PrecisionRecallAUC(window_size=window_size, pos_val=1)
    cm = metrics.ConfusionMatrix()
    labeled_buffer = deque(maxlen=window_size)
    delayed_queue = PriorityQueue()
    cache = {}

    eval_y, eval_probs = [], []
    rolling_prauc_vals = []
    pr_k_vals = []
    curve_rows = []
    labeled_count = 0

    def process_feedback(item_id: int):
        nonlocal labeled_count
        cached = cache.pop(item_id)
        p = cached["prob"]
        y = cached["label"]
        thr = cached["threshold"]

        labeled_count += 1
        metric_prauc.update(y, p)
        cm.update(y, 1 if p > thr else 0)
        labeled_buffer.append({"prob": p, "label": y})
        eval_y.append(y)
        eval_probs.append(p)

        cur_prauc = float(metric_prauc.get())
        if cur_prauc > 0.0:
            rolling_prauc_vals.append(cur_prauc)

        if labeled_count % 100 == 0 or labeled_count == len(stream):
            pr_k, _ = compute_pr_at_k(list(labeled_buffer), alpha=alpha)
            if sum(x["label"] for x in labeled_buffer) > 0:
                pr_k_vals.append(pr_k)
            tr = compute_tradeoff(
                int(cm.true_positives(1)),
                int(cm.false_positives(1)),
                int(cm.true_negatives(1)),
                int(cm.false_negatives(1)),
            )
            curve_rows.append({
                "model_id": "M0_Batch_RF",
                "step": labeled_count,
                "rolling_prauc": round(cur_prauc, 6),
                "pr_at_k": round(pr_k, 6),
                "precision": tr["precision"],
                "recall": tr["recall"],
                "fpr": tr["fpr"],
                "f2": tr["f2"],
                "drift_detected": 0,
                "threshold": thr,
            })

    for idx, item in enumerate(stream):
        step = item["step"]
        prob = float(all_probs_rf[idx])
        cache[step] = {"prob": prob, "label": item["label"], "threshold": block_threshold}
        delayed_queue.put((item["due_step"], step))

        while not delayed_queue.empty() and delayed_queue.queue[0][0] <= step:
            _, f_step = delayed_queue.get()
            process_feedback(f_step)

    while not delayed_queue.empty():
        _, f_step = delayed_queue.get()
        process_feedback(f_step)

    tp = int(cm.true_positives(1))
    fp = int(cm.false_positives(1))
    tn = int(cm.true_negatives(1))
    fn = int(cm.false_negatives(1))
    tr = compute_tradeoff(tp, fp, tn, fn)
    overall_ap = float(average_precision_score(eval_y, eval_probs)) if sum(eval_y) > 0 else 0.0
    final_pr_k, k_eval = compute_pr_at_k(list(labeled_buffer), alpha=alpha)

    elapsed = time.perf_counter() - t0
    print(f"  -> Xong M0 trong {elapsed:.1f}s | Overall AP={overall_ap:.4f} | F2={tr['f2']:.4f} | FP={fp}")

    summary = {
        "model_id": "M0_Batch_RF",
        "description": "Batch Random Forest (Static Baseline)",
        "rows": len(stream),
        "frauds": int(sum(eval_y)),
        "overall_ap": round(overall_ap, 6),
        "mean_rolling_prauc": round(float(np.mean(rolling_prauc_vals)) if rolling_prauc_vals else 0.0, 6),
        "final_rolling_prauc": round(float(metric_prauc.get()), 6),
        "mean_pr_at_k": round(float(np.mean(pr_k_vals)) if pr_k_vals else 0.0, 6),
        "final_pr_at_k": round(final_pr_k, 6),
        "k_eval": k_eval,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": tr["precision"],
        "recall": tr["recall"],
        "fpr": tr["fpr"],
        "f1": tr["f1"],
        "f2": tr["f2"],
        "drift_count": 0,
        "recovery_count": 0,
        "total_replays": 0,
        "avg_latency_ms": round(avg_latency_ms, 3),
        "runtime_sec": round(elapsed, 2),
    }
    return summary, curve_rows


def run_arf_variant(
    cfg: dict,
    base_arf,
    stream: list[dict],
    hist_frauds: list[dict],
    seed: int = 42,
    window_size: int = 1000,
    alpha: float = 1.0,
):
    model_id = cfg["model_id"]
    print(f"\n[Ablation] Dang chay {model_id}: {cfg['description']}...")
    t0 = time.perf_counter()

    random.seed(seed)
    np.random.seed(seed)
    model = copy.deepcopy(base_arf)
    set_arf_grace_period(model, 50)

    metric_prauc = metrics.PrecisionRecallAUC(window_size=window_size, pos_val=1)
    cm = metrics.ConfusionMatrix()
    adwin = drift.ADWIN(delta=0.005)
    labeled_buffer = deque(maxlen=window_size)
    delayed_queue = PriorityQueue()
    cache = {}

    replay_buffer = deque(maxlen=100)
    if cfg["enable_replay"]:
        for h in hist_frauds:
            replay_buffer.append(copy.deepcopy(h))

    block_threshold = 0.70
    replay_interval = 10
    drift_recovery_remaining = 0
    drift_count = 0
    recovery_count = 0
    clean_tx_counter = 0
    total_replays = 0

    eval_y, eval_probs = [], []
    rolling_prauc_vals = []
    pr_k_vals = []
    latencies_ms = []
    curve_rows = []
    labeled_count = 0

    def process_feedback(item_id: int):
        nonlocal labeled_count, clean_tx_counter, total_replays
        nonlocal block_threshold, replay_interval, drift_recovery_remaining
        nonlocal drift_count, recovery_count

        cached = cache.pop(item_id)
        x = cached["features"]
        y = cached["label"]
        prob = cached["prob"]
        thr = cached["threshold"]

        labeled_count += 1

        # 1. Danh gia truoc (Test-then-Train) bang dung xac suat & nguong tai thoi diem t
        metric_prauc.update(y, prob)
        cm.update(y, 1 if prob > thr else 0)
        labeled_buffer.append({"prob": prob, "label": y})
        eval_y.append(y)
        eval_probs.append(prob)

        # 2. Hoc truc tuyen (Online Learning)
        w = cfg["fraud_weight"] if y == 1 else 1.0
        arf_learn_one_weighted(model, x, y, w=w)

        # 3. Experience Replay (Uniform / PER / Noise Aug)
        if cfg["enable_replay"]:
            if y == 1:
                err = abs(1.0 - prob)
                replay_buffer.append({
                    "features": x.copy(),
                    "label": 1,
                    "error": err,
                    "priority": max(0.05, err),
                })
            else:
                clean_tx_counter += 1
                if clean_tx_counter % replay_interval == 0 and len(replay_buffer) > 0:
                    buf_list = list(replay_buffer)
                    if cfg["enable_per"]:
                        priorities = [it["priority"] for it in buf_list]
                        tot_p = sum(priorities)
                        chosen = random.choices(buf_list, weights=priorities, k=1)[0] if tot_p > 0 else random.choice(buf_list)
                    else:
                        chosen = random.choice(buf_list)

                    orig_feat = chosen["features"]
                    if cfg["enable_noise"]:
                        std_val = cfg["noise_std"]
                        aug_feat = {}
                        for k_name, v_val in orig_feat.items():
                            noise = random.gauss(0, std_val * max(1.0, abs(v_val))) if k_name == "Amount" else random.gauss(0, std_val)
                            aug_feat[k_name] = round(v_val + noise, 6)
                    else:
                        aug_feat = orig_feat.copy()

                    arf_learn_one_weighted(model, aug_feat, 1, w=cfg["replay_weight"])
                    total_replays += 1

        # 4. ADWIN tren chuoi sai so tuc thoi e_t = |y_t - p_t| + 3 hanh dong thich ung (chi bat o M6)
        drift_flag = False
        if cfg["enable_adwin_action"]:
            error_t = abs(float(y) - float(prob))
            adwin.update(error_t)
            if adwin.drift_detected:
                drift_count += 1
                drift_flag = True
                drift_recovery_remaining = 500
                set_arf_grace_period(model, 20)
                replay_interval = 3
                block_threshold = 0.55
            elif drift_recovery_remaining > 0:
                drift_recovery_remaining -= 1
                if drift_recovery_remaining == 0:
                    set_arf_grace_period(model, 50)
                    replay_interval = 10
                    block_threshold = 0.70
                    recovery_count += 1

        cur_prauc = float(metric_prauc.get())
        if cur_prauc > 0.0:
            rolling_prauc_vals.append(cur_prauc)

        if labeled_count % 100 == 0 or drift_flag or labeled_count == len(stream):
            pr_k, _ = compute_pr_at_k(list(labeled_buffer), alpha=alpha)
            if labeled_count % 100 == 0 and sum(it["label"] for it in labeled_buffer) > 0:
                pr_k_vals.append(pr_k)
            tr = compute_tradeoff(
                int(cm.true_positives(1)),
                int(cm.false_positives(1)),
                int(cm.true_negatives(1)),
                int(cm.false_negatives(1)),
            )
            curve_rows.append({
                "model_id": model_id,
                "step": labeled_count,
                "rolling_prauc": round(cur_prauc, 6),
                "pr_at_k": round(pr_k, 6),
                "precision": tr["precision"],
                "recall": tr["recall"],
                "fpr": tr["fpr"],
                "f2": tr["f2"],
                "drift_detected": 1 if drift_flag else 0,
                "threshold": thr,
            })

    total_rows = len(stream)
    for item in stream:
        step = item["step"]
        x = item["features"]

        # Luong 1: Du doan tuc thoi (In-flight)
        ts = time.perf_counter()
        proba = model.predict_proba_one(x)
        prob = float(proba.get(1, 0.0)) if proba else 0.0
        latencies_ms.append((time.perf_counter() - ts) * 1000.0)

        cache[step] = {
            "features": x,
            "label": item["label"],
            "prob": prob,
            "threshold": block_threshold,
        }
        delayed_queue.put((item["due_step"], step))

        # Luong 2: Phat cac nhan tre da den han
        while not delayed_queue.empty() and delayed_queue.queue[0][0] <= step:
            _, f_step = delayed_queue.get()
            process_feedback(f_step)

        if step % 5000 == 0:
            print(f"  -> {model_id}: {step:,}/{total_rows:,} giao dich (elapsed={time.perf_counter() - t0:.1f}s)")

    # Xa not hang doi tre cuoi phien
    while not delayed_queue.empty():
        _, f_step = delayed_queue.get()
        process_feedback(f_step)

    tp = int(cm.true_positives(1))
    fp = int(cm.false_positives(1))
    tn = int(cm.true_negatives(1))
    fn = int(cm.false_negatives(1))
    tr = compute_tradeoff(tp, fp, tn, fn)
    overall_ap = float(average_precision_score(eval_y, eval_probs)) if sum(eval_y) > 0 else 0.0
    final_pr_k, k_eval = compute_pr_at_k(list(labeled_buffer), alpha=alpha)
    avg_lat = float(np.mean(latencies_ms)) if latencies_ms else 0.0

    elapsed = time.perf_counter() - t0
    print(
        f"  -> Xong {model_id} trong {elapsed:.1f}s | "
        f"Overall AP={overall_ap:.4f} | Mean PR-AUC={np.mean(rolling_prauc_vals) if rolling_prauc_vals else 0:.4f} | "
        f"Prec={tr['precision']:.4f} | Rec={tr['recall']:.4f} | FPR={tr['fpr']:.4f} | F2={tr['f2']:.4f} | "
        f"Drift={drift_count} (Recovery={recovery_count})"
    )

    summary = {
        "model_id": model_id,
        "description": cfg["description"],
        "rows": total_rows,
        "frauds": int(sum(eval_y)),
        "overall_ap": round(overall_ap, 6),
        "mean_rolling_prauc": round(float(np.mean(rolling_prauc_vals)) if rolling_prauc_vals else 0.0, 6),
        "final_rolling_prauc": round(float(metric_prauc.get()), 6),
        "mean_pr_at_k": round(float(np.mean(pr_k_vals)) if pr_k_vals else 0.0, 6),
        "final_pr_at_k": round(final_pr_k, 6),
        "k_eval": k_eval,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": tr["precision"],
        "recall": tr["recall"],
        "fpr": tr["fpr"],
        "f1": tr["f1"],
        "f2": tr["f2"],
        "drift_count": drift_count,
        "recovery_count": recovery_count,
        "total_replays": total_replays,
        "avg_latency_ms": round(avg_lat, 3),
        "runtime_sec": round(elapsed, 2),
    }
    return summary, curve_rows


def main():
    parser = argparse.ArgumentParser(description="Chay thuc nghiem doi chung Ablation Study (M0-M6)")
    parser.add_argument("--rows", type=int, default=30000, help="So giao dich kiem thu (mac dinh: 30000)")
    parser.add_argument("--drift-start", type=int, default=10000, help="Buoc bat dau Concept Drift (mac dinh: 10000)")
    parser.add_argument("--drift-len", type=int, default=3000, help="Do dai dot Concept Drift (mac dinh: 3000)")
    parser.add_argument("--no-drift", action="store_true", help="Tat Concept Drift neu muon chay luong tinh")
    parser.add_argument("--window-size", type=int, default=1000, help="Cua so truot W (mac dinh: 1000)")
    parser.add_argument("--alpha", type=float, default=1.0, help="He so Alert Budget alpha (mac dinh: 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed co dinh (mac dinh: 42)")
    parser.add_argument("--out-dir", type=str, default="results", help="Thu muc luu file CSV ket qua")
    args = parser.parse_args()

    if not os.path.exists(MODEL_ARF_FILE) or not os.path.exists(MODEL_RF_FILE):
        print("[Error] Thieu model_arf.pkl hoac model_rf.pkl. Hay chay python pretrain.py truoc!")
        return

    print(f"[Setup] Nap mo hinh pretrained tu {MODEL_ARF_FILE} va {MODEL_RF_FILE}...")
    with open(MODEL_ARF_FILE, "rb") as f:
        base_arf = pickle.load(f)
    with open(MODEL_RF_FILE, "rb") as f:
        model_rf = pickle.load(f)
    if hasattr(model_rf, "named_steps") and "randomforestclassifier" in model_rf.named_steps:
        model_rf.named_steps["randomforestclassifier"].n_jobs = 1

    hist_frauds = load_historical_frauds(PREFIX_FILE, max_items=100)
    print(f"[Setup] Da nap {len(hist_frauds)} mau gian lan lich su cho Replay Buffer.")

    drift_enabled = not args.no_drift
    print(
        f"[Setup] Dang dung luong dung chung: {args.rows:,} giao dich | "
        f"Drift={drift_enabled} (buoc {args.drift_start:,} -> {args.drift_start + args.drift_len:,}) | Seed={args.seed}"
    )
    stream = build_shared_stream(
        data_path=SUFFIX_FILE,
        rows=args.rows,
        drift_enabled=drift_enabled,
        drift_start=args.drift_start,
        drift_len=args.drift_len,
        seed=args.seed,
    )
    total_frauds = sum(item["label"] for item in stream)
    print(f"[Setup] Luong san sang: {len(stream):,} giao dich | Tong gian lan: {total_frauds:,} ({total_frauds / len(stream):.2%})")

    configs = [
        {
            "model_id": "M1_ARF_Pure",
            "description": "ARF thuan tuy (w=1, No Replay, No Noise, No ADWIN Action)",
            "fraud_weight": 1.0,
            "replay_weight": 1.0,
            "enable_replay": False,
            "enable_per": False,
            "enable_noise": False,
            "noise_std": 0.0,
            "enable_adwin_action": False,
        },
        {
            "model_id": "M2_ARF_Weight250",
            "description": "ARF + Class Weight 250 (Khong Replay)",
            "fraud_weight": 250.0,
            "replay_weight": 1.0,
            "enable_replay": False,
            "enable_per": False,
            "enable_noise": False,
            "noise_std": 0.0,
            "enable_adwin_action": False,
        },
        {
            "model_id": "M3_ARF_UniformReplay",
            "description": "ARF + Uniform Replay (Lay mau deu)",
            "fraud_weight": 1.0,
            "replay_weight": 1.0,
            "enable_replay": True,
            "enable_per": False,
            "enable_noise": False,
            "noise_std": 0.0,
            "enable_adwin_action": False,
        },
        {
            "model_id": "M4_ARF_PER",
            "description": "ARF + PER (Khong nhieu Gaussian)",
            "fraud_weight": 1.0,
            "replay_weight": 1.0,
            "enable_replay": True,
            "enable_per": True,
            "enable_noise": False,
            "noise_std": 0.0,
            "enable_adwin_action": False,
        },
        {
            "model_id": "M5_ARF_PER_Noise",
            "description": "ARF + PER + Gaussian Noise (sigma=0.02)",
            "fraud_weight": 1.0,
            "replay_weight": 1.0,
            "enable_replay": True,
            "enable_per": True,
            "enable_noise": True,
            "noise_std": 0.02,
            "enable_adwin_action": False,
        },
        {
            "model_id": "M6_ARF_Full_ADWIN",
            "description": "ARF + PER + Noise + ADWIN Adaptive Action",
            "fraud_weight": 1.0,
            "replay_weight": 1.0,
            "enable_replay": True,
            "enable_per": True,
            "enable_noise": True,
            "noise_std": 0.02,
            "enable_adwin_action": True,
        },
    ]

    summaries = []
    all_curves = []

    # Chay M0 (Batch RF) truoc
    s0, c0 = run_batch_rf(model_rf, stream, window_size=args.window_size, alpha=args.alpha)
    summaries.append(s0)
    all_curves.extend(c0)

    # Chay lan luot M1 -> M6
    for cfg in configs:
        s, c = run_arf_variant(
            cfg=cfg,
            base_arf=base_arf,
            stream=stream,
            hist_frauds=hist_frauds,
            seed=args.seed,
            window_size=args.window_size,
            alpha=args.alpha,
        )
        summaries.append(s)
        all_curves.extend(c)

    # Luu ket qua CSV
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "ablation_summary.csv")
    curve_path = os.path.join(args.out_dir, "ablation_curve.csv")

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    with open(curve_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_curves[0].keys()))
        writer.writeheader()
        writer.writerows(all_curves)

    print("\n" + "=" * 110)
    print("BANG TONG HOP ABLATION STUDY (M0 - M6)")
    print("=" * 110)
    header = f"{'Model':<22} | {'All-AP':>7} | {'MeanPR':>7} | {'Pr@k':>7} | {'Prec':>7} | {'Recall':>7} | {'FPR':>7} | {'F2':>7} | {'FP':>5} | {'Drift':>5} | {'ms':>6}"
    print(header)
    print("-" * 110)
    for s in summaries:
        print(
            f"{s['model_id']:<22} | {s['overall_ap']:>7.4f} | {s['mean_rolling_prauc']:>7.4f} | "
            f"{s['mean_pr_at_k']:>7.4f} | {s['precision']:>7.4f} | {s['recall']:>7.4f} | "
            f"{s['fpr']:>7.4f} | {s['f2']:>7.4f} | {s['fp']:>5d} | {s['drift_count']:>5d} | {s['avg_latency_ms']:>6.3f}"
        )
    print("=" * 110)
    print(f"[Done] Da luu bang tong hop tai: {summary_path}")
    print(f"[Done] Da luu du lieu do thi tai: {curve_path}")


if __name__ == "__main__":
    main()