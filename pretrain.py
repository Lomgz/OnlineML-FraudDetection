import csv
import pickle
import time
from river import forest, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline

DATA_FILE = "data/creditcard_prefix.csv"
MODEL_HAT_OUTPUT = "model_hat.pkl"
MODEL_ARF_OUTPUT = "model_arf.pkl"
MODEL_RF_OUTPUT  = "model_rf.pkl"
# Tuong thich nguoc:
MODEL_SGD_OUTPUT = "model_sgd.pkl"
MODEL_ADAM_OUTPUT = "model_adam.pkl"


def parse_row(row: dict) -> tuple[dict, int]:
    label = int(float(row.pop("Class", 0)))
    row.pop("Time", None)
    features = {k: float(v) for k, v in row.items() if v != ""}
    return features, label


def main():
    print(f"[Pretrain] Bat dau huan luyen tu tep: {DATA_FILE}")

    # 1. Khoi tao mo hinh streaming chuyen biet (HAT & ARF)
    model_hat = tree.HoeffdingAdaptiveTreeClassifier(grace_period=50, seed=42)
    model_arf = forest.ARFClassifier(n_models=15, grace_period=50, seed=42)

    count = 0
    fraud_count = 0
    start_time = time.time()
    X_train_batch, y_train_batch = [], []

    try:
        with open(DATA_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for raw_row in reader:
                features, label = parse_row(raw_row)

                # Trong so phat cao cho HAT; River ARFClassifier.learn_one bo qua **kwargs
                # va tu dung Poisson bagging, nen goi truc tiep learn_one(features, label).
                weight = 250.0 if label == 1 else 1.0
                model_hat.learn_one(features, label, w=weight)
                model_arf.learn_one(features, label)

                X_train_batch.append(features)
                y_train_batch.append(label)

                if label == 1:
                    fraud_count += 1
                count += 1

                if count % 20000 == 0:
                    print(f"  -> Da huan luyen {count:,} giao dich...")

    except FileNotFoundError:
        print(f"[Pretrain Error] Khong tim thay tep: {DATA_FILE}")
        return

    # 2. Huan luyen Batch Random Forest lam baseline tinh
    print("[Pretrain] Huan luyen Batch Random Forest (Scikit-Learn baseline)...")
    model_rf = make_pipeline(
        DictVectorizer(sparse=False),
        RandomForestClassifier(n_estimators=50, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
    )
    model_rf.fit(X_train_batch, y_train_batch)

    # 3. Luu tru cac mo hinh
    for path, model in [
        (MODEL_HAT_OUTPUT, model_hat),
        (MODEL_ARF_OUTPUT, model_arf),
        (MODEL_RF_OUTPUT, model_rf),
        (MODEL_SGD_OUTPUT, model_hat),
        (MODEL_ADAM_OUTPUT, model_arf),
    ]:
        with open(path, "wb") as f:
            pickle.dump(model, f)

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"[Pretrain] Hoan tat: {count:,} giao dich ({fraud_count} gian lan) trong {elapsed:.1f}s.")
    print(f"[Pretrain] Da luu: {MODEL_HAT_OUTPUT}, {MODEL_ARF_OUTPUT}, {MODEL_RF_OUTPUT}.")


if __name__ == "__main__":
    main()
