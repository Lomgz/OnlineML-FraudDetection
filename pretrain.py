import csv
import pickle
import time
from river import linear_model, preprocessing, compose, optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline

DATA_FILE = "data/creditcard_prefix.csv"
MODEL_SGD_OUTPUT = "model_sgd.pkl"
MODEL_ADAM_OUTPUT = "model_adam.pkl"

def parse_row(row: dict) -> tuple[dict, int]:
    label = int(float(row.pop("Class", 0)))
    row.pop("Time", None) # Remove Time column
    features = {k: float(v) for k, v in row.items() if v != ""}
    return features, label

def main():
    print(f"[PRETRAIN] Starting Pre-training process from file: {DATA_FILE}")
    
    # 1. Initialize empty models
    opt_sgd = optim.SGD(lr=0.1)
    model_sgd = compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(optimizer=opt_sgd)
    )

    opt_adam = optim.Adam(lr=0.01)
    model_adam = compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(optimizer=opt_adam)
    )

    count = 0
    fraud_count = 0
    start_time = time.time()

    # 2. Train with historical data
    X_train_batch = []
    y_train_batch = []

    # 2. Train with historical data
    try:
        with open(DATA_FILE, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            
            for raw_row in reader:
                features, label = parse_row(raw_row)
                
                # Cập nhật từng dòng cho mô hình Online (SGD, Adam)
                model_sgd.learn_one(features, label)
                model_adam.learn_one(features, label)
                
                # Gom dữ liệu lại cho mô hình Batch (Random Forest)
                X_train_batch.append(features)
                y_train_batch.append(label)
                
                if label == 1:
                    fraud_count += 1
                
                count += 1
                if count % 10000 == 0:
                    print(f"[PRETRAIN] Trained {count} transactions...")
                    
    except FileNotFoundError:
        print(f"[PRETRAIN] Error: File {DATA_FILE} not found")
        return

    print("[PRETRAIN] Training Batch Random Forest model...")
    model_rf = make_pipeline(
        DictVectorizer(sparse=False),
        RandomForestClassifier(n_estimators=50, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    )
    model_rf.fit(X_train_batch, y_train_batch)

    # 3. Save the model to .pkl file
    with open(MODEL_SGD_OUTPUT, "wb") as f:
        pickle.dump(model_sgd, f)
    with open(MODEL_ADAM_OUTPUT, "wb") as f:
        pickle.dump(model_adam, f)
    with open("model_rf.pkl", "wb") as f:
        pickle.dump(model_rf, f)
        
    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"[PRETRAIN] Completed training {count} transactions ({fraud_count} frauds).")
    print(f"[PRETRAIN] Saved SGD model to: {MODEL_SGD_OUTPUT}")
    print(f"[PRETRAIN] Saved Adam model to: {MODEL_ADAM_OUTPUT}")
    print(f"[PRETRAIN] Saved RF model to: model_rf.pkl")

    # 3. Save the model to .pkl file
    with open(MODEL_SGD_OUTPUT, "wb") as f:
        pickle.dump(model_sgd, f)
    with open(MODEL_ADAM_OUTPUT, "wb") as f:
        pickle.dump(model_adam, f)
    with open("model_rf.pkl", "wb") as f:
        pickle.dump(model_rf, f)
        
    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"[PRETRAIN] Completed training {count} transactions ({fraud_count} frauds).")
    print(f"[PRETRAIN] Time taken: {elapsed:.2f} seconds.")
    print(f"[PRETRAIN] Saved SGD model to: {MODEL_SGD_OUTPUT}")
    print(f"[PRETRAIN] Saved Adam model to: {MODEL_ADAM_OUTPUT}")
    print(f"[PRETRAIN] Saved RF model to: model_rf.pkl")
    print("[PRETRAIN] The Server can now load these files for immediate use!")

if __name__ == "__main__":
    main()