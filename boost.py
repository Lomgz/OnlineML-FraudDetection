import argparse
import csv
import json
import os
import sys
import time
import uuid
import requests
from requests.adapters import HTTPAdapter

parser = argparse.ArgumentParser(description="Turbo Stream Simulator - Ban luong giao dich toc do cao")
parser.add_argument("--rows", type=int, default=5000, help="So giao dich can gui, mac dinh: 5000")
parser.add_argument("--server", type=str, default="http://localhost:8000", help="Dia chi ML Engine server")
parser.add_argument("--batch-size", type=int, default=50, help="Kich thuoc micro-batch (mac dinh: 50)")
parser.add_argument("--drift", action="store_true", help="Kich hoat Concept Drift")
parser.add_argument("--drift-start", type=int, default=1500, help="Giao dich bat dau kich hoat Drift")
parser.add_argument("--drift-len", type=int, default=800, help="Do dai dot tan cong Concept Drift")
parser.add_argument("--data", type=str, default="data/creditcard_suffix.csv", help="Duong dan tep CSV giao dich")
args = parser.parse_args()

SERVER = args.server.rstrip("/")
BATCH_URL = f"{SERVER}/batch_stream"
STREAM_TX_URL = f"{SERVER}/stream_tx"
PREDICT_URL = f"{SERVER}/predict"
FEEDBACK_URL = f"{SERVER}/feedback"
STATUS_FILE = "boost_status.json"
PID_FILE = "boost.pid"


def update_status(count: int, total: int, fraud_count: int, running: bool, speed: float = 0.0):
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "running": running,
                "count": count,
                "total": total,
                "fraud_count": fraud_count,
                "speed": round(speed, 1),
                "timestamp": round(time.time(), 2)
            }, f)
    except Exception:
        pass


def parse_row(row: dict) -> tuple[dict, int]:
    label = int(float(row.pop("Class", 0)))
    row.pop("Time", None)
    features = {k: float(v) for k, v in row.items() if v != ""}
    return features, label


def main():
    try:
        with open(PID_FILE, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except Exception:
        pass

    print(f"[Turbo Boost] Bat dau ban luong toc do cao toi: {SERVER}")
    print(f"[Turbo Boost] Muc tieu: {args.rows:,} giao dich | Batch size: {args.batch_size} | Drift: {args.drift}")
    print("-" * 60)

    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=1)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        health = session.get(SERVER, timeout=3)
        print(f"[Turbo Boost] Trang thai server: {health.json().get('status', 'unknown')}")
    except Exception:
        print("[Turbo Boost Error] Khong the ket noi toi server. Vui long bat server truoc!")
        update_status(0, args.rows, 0, False)
        if os.path.exists(PID_FILE):
            try:
                os.remove(PID_FILE)
            except Exception:
                pass
        sys.exit(1)

    count = 0
    fraud_count = 0
    start_time = time.time()
    update_status(0, args.rows, 0, True, 0.0)

    use_batch = True
    try:
        test_resp = session.post(BATCH_URL, json={"transactions": []}, timeout=2)
        if test_resp.status_code != 200:
            use_batch = False
    except Exception:
        use_batch = False

    mode_str = f"Batch Micro-Stream ({args.batch_size} txs/req)" if use_batch else "Single Stream"
    print(f"[Turbo Boost] Che do truyen: {mode_str}")

    try:
        with open(args.data, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            batch = []

            for raw_row in reader:
                if count >= args.rows:
                    break

                features, true_label = parse_row(raw_row)

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
                if true_label == 1:
                    fraud_count += 1
                count += 1

                if use_batch:
                    batch.append({"features": features, "label": true_label, "transaction_id": tx_id})
                    if len(batch) >= args.batch_size or count >= args.rows:
                        try:
                            session.post(BATCH_URL, json={"transactions": batch}, timeout=5)
                        except Exception as e:
                            print(f"[Turbo Boost] Loi gui batch: {e}")
                        batch = []
                else:
                    try:
                        session.post(STREAM_TX_URL, json={"features": features, "label": true_label, "transaction_id": tx_id}, timeout=3)
                    except Exception:
                        try:
                            session.post(PREDICT_URL, json={"features": features, "transaction_id": tx_id}, timeout=3)
                            session.post(FEEDBACK_URL, json={"features": features, "label": true_label, "transaction_id": tx_id}, timeout=3)
                        except Exception:
                            pass

                if count % 200 == 0 or count >= args.rows:
                    elapsed = max(0.01, time.time() - start_time)
                    cur_speed = count / elapsed
                    update_status(count, args.rows, fraud_count, True, cur_speed)
                    if count % 1000 == 0:
                        print(f"[Turbo Boost] Da gui {count:,}/{args.rows:,} giao dich ({cur_speed:.0f} tx/s, {fraud_count} gian lan)")

    except FileNotFoundError:
        print(f"[Turbo Boost Error] Khong tim thay tep: {args.data}")
    except KeyboardInterrupt:
        print("\n[Turbo Boost] Da dung boi nguoi dung.")
    except Exception as e:
        print(f"[Turbo Boost Error] Loi ngoai y muon: {e}")
    finally:
        elapsed = max(0.01, time.time() - start_time)
        final_speed = count / elapsed
        update_status(count, args.rows, fraud_count, False, final_speed)
        if os.path.exists(PID_FILE):
            try:
                os.remove(PID_FILE)
            except Exception:
                pass
        print("-" * 60)
        print(f"[Turbo Boost] Hoan tat: {count:,} giao dich ({fraud_count} gian lan)")
        print(f"[Turbo Boost] Tong thoi gian: {elapsed:.2f}s | Toc do trung binh: {final_speed:.0f} tx/s")


if __name__ == "__main__":
    main()
