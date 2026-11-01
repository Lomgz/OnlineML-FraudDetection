import argparse
import csv
import sys
import time
import uuid
import requests

parser = argparse.ArgumentParser(description="Payment Edge Gateway - Gia lap luong giao dich truc tuyen")
parser.add_argument("--delay", type=float, default=0.1, help="Do tre giua cac giao dich (giay), mac dinh: 0.1")
parser.add_argument("--rows", type=int, default=1000, help="So giao dich toi da can gui, mac dinh: 1000")
parser.add_argument("--server", type=str, default="http://localhost:8000", help="Dia chi ML Engine server")
parser.add_argument("--feedback-delay", type=float, default=0.1, help="Do tre truoc khi gui nhan phan hoi (giay)")
parser.add_argument("--drift", action="store_true", help="Kich hoat Concept Drift (bien dong hanh vi gian lan)")
parser.add_argument("--drift-start", type=int, default=1500, help="Giao dich bat dau kich hoat Drift")
parser.add_argument("--drift-len", type=int, default=800, help="Do dai dot tan cong Concept Drift")
parser.add_argument("--data", type=str, default="data/creditcard_suffix.csv", help="Duong dan tep CSV giao dich")
args = parser.parse_args()

SERVER = args.server.rstrip("/")
PREDICT_URL = f"{SERVER}/predict"
FEEDBACK_URL = f"{SERVER}/feedback"


def parse_row(row: dict) -> tuple[dict, int]:
    label = int(float(row.pop("Class", 0)))
    row.pop("Time", None)
    features = {k: float(v) for k, v in row.items() if v != ""}
    return features, label


def send_predict(features: dict, tx_id: str) -> dict:
    resp = requests.post(PREDICT_URL, json={"features": features, "transaction_id": tx_id}, timeout=5)
    resp.raise_for_status()
    return resp.json()


def send_feedback(features: dict, label: int, tx_id: str) -> dict:
    resp = requests.post(FEEDBACK_URL, json={"features": features, "label": label, "transaction_id": tx_id}, timeout=5)
    resp.raise_for_status()
    return resp.json()


def main():
    print(f"[Payment Edge] Ket noi toi server: {SERVER}")
    print(f"[Payment Edge] Tep du lieu: {args.data} | Muc tieu: {args.rows} giao dich | Do tre: {args.delay}s")
    print("-" * 60)

    try:
        health = requests.get(SERVER, timeout=3)
        print(f"[Payment Edge] Trang thai server: {health.json().get('status', 'unknown')}")
    except Exception:
        print("[Payment Edge Error] Khong the ket noi toi server. Vui long bat server.py truoc!")
        sys.exit(1)

    count = 0
    fraud_count = 0
    blocked_count = 0

    try:
        with open(args.data, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)

            for raw_row in reader:
                if count >= args.rows:
                    break

                features, true_label = parse_row(raw_row)

                # Gia lap Concept Drift neu duoc bat
                if args.drift and args.drift_start <= count < (args.drift_start + args.drift_len):
                    if true_label == 1:
                        for k in ["V10", "V12", "V14", "V17"]:
                            if k in features:
                                features[k] = -features[k] * 1.5
                    elif count % 10 == 0:
                        true_label = 1
                        for k in ["V4", "V11"]:
                            if k in features:
                                features[k] = -features[k]

                tx_id = str(uuid.uuid4())[:8]

                # 1. Du doan (Inference tai Edge Gateway)
                try:
                    result = send_predict(features, tx_id)
                    prob = result.get("fraud_probability", 0.0)
                    action = result.get("action", "?")

                    status_str = "GIAN LAN" if true_label == 1 else "Hop le  "
                    action_str = "CHAN  " if action == "BLOCK" else "CHO QUA"
                    print(f"[{count+1:>4}] TX-{tx_id} | {status_str} | P(fraud)={prob:.4f} | Quyet dinh: {action_str}")

                    if action == "BLOCK":
                        blocked_count += 1
                    if true_label == 1:
                        fraud_count += 1
                except Exception as e:
                    print(f"[{count+1:>4}] TX-{tx_id} | Loi du doan: {e}")
                    count += 1
                    continue

                # 2. Phan hoi nhan tre (Delayed Feedback)
                time.sleep(args.feedback_delay)
                try:
                    fb_result = send_feedback(features, true_label, tx_id)
                    prauc = fb_result.get("prauc_arf", fb_result.get("prauc_sgd", 0.0))
                    pr_k = fb_result.get("pr_at_k_arf", fb_result.get("pr_at_k_sgd", 0.0))
                    k_val = fb_result.get("k_eval", 1)
                    print(f"       -> Feedback: Nhan={true_label} | PR-AUC={prauc:.4f} | Pr@{k_val}={pr_k:.2%}")
                except Exception as e:
                    print(f"       -> Feedback loi: {e}")

                count += 1
                time.sleep(args.delay)

    except FileNotFoundError:
        print(f"[Payment Edge Error] Khong tim thay tep: {args.data}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[Payment Edge] Da dung boi nguoi dung.")

    print("-" * 60)
    print(f"[Payment Edge] Hoan tat: {count} giao dich | {fraud_count} gian lan thuc te | {blocked_count} da chan")


if __name__ == "__main__":
    main()
