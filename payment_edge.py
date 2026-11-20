import argparse
import csv
import sys
import time
import uuid
import requests
import numpy as np
from queue import PriorityQueue

parser = argparse.ArgumentParser(description="Payment Edge Gateway - Gia lap luong giao dich truc tuyen")
parser.add_argument("--delay", type=float, default=0.1, help="Do tre giua cac giao dich (giay), mac dinh: 0.1")
parser.add_argument("--rows", type=int, default=1000, help="So giao dich toi da can gui, mac dinh: 1000")
parser.add_argument("--server", type=str, default="http://localhost:8000", help="Dia chi ML Engine server")
parser.add_argument("--feedback-delay", type=float, default=0.1, help="Do tre truoc khi gui nhan phan hoi (giay)")
parser.add_argument("--drift", action="store_true", help="Kich hoat Concept Drift (bien dong hanh vi gian lan)")
parser.add_argument("--drift-start", type=int, default=1500, help="Giao dich bat dau kich hoat Drift")
parser.add_argument("--drift-len", type=int, default=800, help="Do dai dot tan cong Concept Drift")
parser.add_argument("--data", type=str, default="data/creditcard_suffix.csv", help="Duong dan tep CSV giao dich")
parser.add_argument("--seed", type=int, default=42, help="Random seed cho phan phoi do tre")
args = parser.parse_args()

SERVER = args.server.rstrip("/")
PREDICT_URL = f"{SERVER}/predict"
FEEDBACK_URL = f"{SERVER}/feedback"
session = requests.Session()

def parse_row(row: dict) -> tuple[dict, int]:
    label = int(float(row.pop("Class", 0)))
    row.pop("Time", None)
    features = {k: float(v) for k, v in row.items() if v != ""}
    return features, label


def send_predict(features: dict, tx_id: str) -> dict:
    resp = session.post(PREDICT_URL, json={"features": features, "transaction_id": tx_id}, timeout=5)
    resp.raise_for_status()
    return resp.json()


def send_feedback(label: int, tx_id: str) -> dict:
    resp = session.post(FEEDBACK_URL, json={"label": label, "transaction_id": tx_id}, timeout=5)
    resp.raise_for_status()
    return resp.json()

def main():
    np.random.seed(args.seed)
    print(f"[Payment Edge] Ket noi toi server: {SERVER}")
    print(f"[Payment Edge] Tep du lieu: {args.data} | Muc tieu: {args.rows} giao dich | Do tre: {args.delay}s")
    print("-" * 60)

    try:
        health = session.get(SERVER, timeout=3)
        print(f"[Payment Edge] Trang thai server: {health.json().get('status', 'unknown')}")
    except Exception:
        print("[Payment Edge Error] Khong the ket noi toi server. Vui long bat server.py truoc!")
        sys.exit(1)

    count = 0
    fraud_count = 0
    blocked_count = 0
    seq_id = 0
    
    # Hàng đợi ưu tiên lưu các nhãn bị trễ: (due_time, seq_id, transaction_id, true_label)
    delayed_queue = PriorityQueue()
    
    # Đồng hồ thời gian logic (mỗi giao dịch tính là 1 tick)
    current_time_tick = 0

    try:
        with open(args.data, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)

            for raw_row in reader:
                if count >= args.rows:
                    break
                
                current_time_tick += 1
                features, true_label = parse_row(raw_row)

                # Giả lập Concept Drift nếu được bật
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

                tx_id = f"{current_time_tick}-{uuid.uuid4().hex[:6]}"

                # ==========================================
                # LUỒNG 1: DỰ ĐOÁN TỨC THỜI (IN-FLIGHT)
                # ==========================================
                try:
                    result = send_predict(features, tx_id)
                    prob = result.get("fraud_probability", 0.0)
                    action = result.get("action", "?")
                    lat_arf_ms = result.get("lat_arf_ms", result.get("processing_time_ms", 0.0))

                    status_str = "GIAN LAN" if true_label == 1 else "Hop le  "
                    action_str = "CHAN  " if action == "BLOCK" else "CHO QUA"
                    print(f"[{current_time_tick:>4}] TX-{tx_id} | {status_str} | P(fraud)={prob:.4f} | ARF={lat_arf_ms:.2f}ms | Quyet dinh: {action_str}")

                    if action == "BLOCK":
                        blocked_count += 1
                    if true_label == 1:
                        fraud_count += 1
                except Exception as e:
                    print(f"[{current_time_tick:>4}] TX-{tx_id} | Loi du doan: {e}")
                    count += 1
                    continue

                # ==========================================
                # TẠO ĐỘ TRỄ CHO NHÃN VÀ ĐƯA VÀO HÀNG ĐỢI
                # ==========================================
                if true_label == 0:
                    # Giao dịch hợp lệ: Trễ trung bình 50 bước, dao động 10
                    delay_steps = int(max(1, round(np.random.normal(50, 10))))
                else:
                    # Giao dịch gian lận: Trễ sâu theo Chargeback (Log-Normal, trung bình ~400 bước)
                    delay_steps = int(max(1, round(np.random.lognormal(6, 0.5))))
                
                due_time = current_time_tick + delay_steps
                seq_id += 1
                delayed_queue.put((due_time, seq_id, tx_id, true_label))

                # ==========================================
                # LUỒNG 2: PHÁT NHÃN TRỄ VỀ SERVER
                # ==========================================
                # Lấy tất cả các nhãn đã "chín" (due_time <= current_time_tick)
                while not delayed_queue.empty():
                    earliest_due, _, f_tx_id, f_label = delayed_queue.queue[0]
                    if earliest_due <= current_time_tick:
                        delayed_queue.get() # Lấy hẳn ra khỏi queue
                        try:
                            fb_result = send_feedback(f_label, f_tx_id)
                            prauc = fb_result.get("prauc_arf", 0.0)
                            pr_k = fb_result.get("pr_at_k_arf", 0.0)
                            k_val = fb_result.get("k_eval", 1)
                            drift_tag = " [DRIFT!]" if fb_result.get("drift_detected") else ""
                            print(f"       -> [TRỄ] Feedback TX-{f_tx_id}: Nhan={f_label} | PR-AUC={prauc:.4f} | Pr@{k_val}={pr_k:.2%}{drift_tag}")
                        except Exception as e:
                            print(f"       -> [TRỄ] Feedback loi TX-{f_tx_id}: {e}")
                    else:
                        break # Nếu cái sớm nhất chưa tới hạn, các cái sau cũng chưa tới

                count += 1
                if args.delay > 0:
                    time.sleep(args.delay)

            # (Tùy chọn) Sau khi hết data, dọn sạch hàng đợi còn dư
            print(f"\n[Payment Edge] Dang cho cac nhan tre cuoi cung (Con {delayed_queue.qsize()} nhan)...")
            while not delayed_queue.empty():
                earliest_due, _, f_tx_id, f_label = delayed_queue.get()
                try:
                    send_feedback(f_label, f_tx_id)
                except Exception as e:
                    print(f"       -> [FLUSH LOI] TX-{f_tx_id}: {e}")

    except FileNotFoundError:
        print(f"[Payment Edge Error] Khong tim thay tep: {args.data}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[Payment Edge] Da dung boi nguoi dung.")

    print("-" * 60)
    print(f"[Payment Edge] Hoan tat: {count} giao dich | {fraud_count} gian lan thuc te | {blocked_count} da chan")


if __name__ == "__main__":
    main()
