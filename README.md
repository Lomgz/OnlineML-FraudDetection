# Payment Edge & Streaming Broker Fraud Detection Engine

A high-throughput, sub-millisecond **Online Machine Learning (Streaming ML)** system designed for **Payment Edge Gateways** and **Cloud Message Broker Architectures (Apache Kafka / Apache Flink)**.

---

## 1. System Positioning & Core Thesis

Traditional batch-learning models (e.g., XGBoost or Random Forest retrained on daily/weekly schedules) suffer from critical limitations in high-velocity financial payment environments:
1. **Blind Window & Concept Drift Lag:** New fraud patterns bypass static models during the multi-hour or multi-day lag between batch retraining cycles.
2. **Inference Latency & Heavy Footprint:** Traditional ensemble models demand high compute/memory, making them unsuitable for in-flight transaction authorization at network edges.

This engine repositions fraud detection into **two complementary production deployment topologies**:

```
+--------------------------------------------------------------------------------------------------+
|                                    DEPLOYMENT TOPOLOGIES                                         |
+--------------------------------------------------------------------------------------------------+

  TOPOLOGY A: Payment Edge Gateway (In-line Low-Latency Scoring)
  +--------------------+         < 5ms Latency         +---------------------------+
  | POS / E-Commerce   | ----------------------------> | Payment Edge Gateway      | ---> (Issuer Bank)
  | Checkout Terminal  | <---------------------------- | (FastAPI + River ML O(1)) |
  +--------------------+       ALLOW / BLOCK           +---------------------------+
                                                                     |
                                                                     | Emits Transaction Events & Feedback
                                                                     v
  TOPOLOGY B: Cloud / Data Center Streaming (Message Broker Decoupled Architecture)
  +------------------------------------------------------------------------------------------------+
  | Message Broker (Apache Kafka / Redis Streams / Apache Flink)                                   |
  |  - Topic `payment-transactions`: High-throughput ingestion & distributed inference             |
  |  - Topic `fraud-chargebacks`: Delayed ground-truth feedback streaming for live weight updates   |
  +------------------------------------------------------------------------------------------------+
                                 |                                   |
                                 v                                   v
                    +---------------------------+       +---------------------------+
                    | Live Streamlit Monitoring |       | Cold Lake / Audit Storage |
                    | (PR-AUC, Pr@k, Drift)     |       | (Compliance & Forensics)  |
                    +---------------------------+       +---------------------------+
```

### Why Online Logistic Regression (SGD vs. Adam) for Edge Gateways?
- **$O(1)$ Memory & Compute Footprint:** Model state requires only storing weight vectors $W$ (and moments $m, v$ for Adam). No raw transactions need to be persisted in memory.
- **Sub-millisecond SLA:** Scoring takes `< 1ms`, fitting comfortably inside payment gateway authorization windows (< 50ms).
- **Instantaneous Weight Adaptation:** The moment a chargeback or merchant confirmation arrives, weights adapt via `learn_one()` without taking the service offline.
- **Drift Responsiveness:** SGD adapts instantly to sudden shifts in fraud strategies without momentum damping, while Adam provides smooth convergence for steady-state traffic.

---

## 2. Experimental Protocol v2 Compliance & Information Parity

To ensure rigorous benchmarking and eliminate data leakage, the dataset is split chronologically into two 50% partitions:
- **`data/creditcard_prefix.csv` (142,403 rows, 269 frauds - 0.189% rate):** Historical warm-up partition used exclusively by `pretrain.py` to overcome the "Cold Start" problem.
- **`data/creditcard_suffix.csv` (142,404 rows, 223 frauds - 0.157% rate):** Strictly locked evaluation stream used exclusively by stream producers (`payment_edge.py`, `boost.py`).

### Scientific Evaluation Metrics (Protocol v2)
In extreme class imbalance (fraud rate < 0.2%), standard ROC-AUC and Accuracy provide overly optimistic, misleading scores. This engine enforces:
- **`PrecisionRecallAUC` (Rolling PRAUC):** Measures the Area Under the Precision-Recall curve over a rolling sliding window ($W=1000$).
- **$Pr@k$ (Precision at $k$):** Measures the precision within the top-$k$ riskiest transactions matching operational alert budgets:
  $$k = \max(1, \text{round}(\alpha \times \text{fraud\_rate} \times M))$$
  where $\alpha = 1.0$ (default alert budget multiplier) and $M$ is the labeled window size.

---

## 3. Project Architecture

The codebase consists of modular, loosely-coupled components:

- **`pretrain.py`**: Pre-trains SGD and Adam online logistic regression pipelines on `creditcard_prefix.csv` and serializes states to `model_sgd.pkl` and `model_adam.pkl`.
- **`server.py`**: The Streaming ML Engine exposing REST endpoints for Edge Gateways and Message Broker consumers:
  - `POST /predict`: Real-time transaction risk scoring (`ALLOW` vs `BLOCK` at threshold 0.7).
  - `POST /feedback`: Ingests delayed labels, executes `learn_one()`, and updates `RollingPRAUC` and $Pr@k$.
  - `GET /stats`: Returns live metrics (`prauc_sgd`, `prauc_adam`, `pr_at_k_sgd`, `pr_at_k_adam`, confusion matrices).
  - `GET /logs`: Serves the latest transaction audit ring buffer.
- **`view.py`**: Streamlit operational dashboard tracking real-time $Pr@k$, PR-AUC, transaction logs, and ALLOW/BLOCK ratios.
- **`payment_edge.py`**: Controlled-rate stream simulator mimicking Payment Edge Gateway traffic feeding into the ML Engine.
- **`boost.py`**: High-performance HTTP session pooling benchmark tool for load and throughput testing.

---

## 4. Getting Started

### 1. Environment Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Pre-training the Edge Models (Prefix 50%)
Initialize the model weights on the 50% historical prefix to eliminate the cold-start penalty:
```bash
python pretrain.py
```
*Outputs: `model_sgd.pkl` and `model_adam.pkl`.*

### 3. Running the Streaming System
Open three separate terminal windows with virtual environment activated:

**Terminal 1: Start the ML Engine (Edge Gateway / Broker Consumer)**
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2: Launch the Operational Dashboard**
```bash
streamlit run view.py
```

**Terminal 3: Start the Payment Stream Simulator (Suffix 50%)**
```bash
# Standard Payment Edge simulation with 10ms pacing
python payment_edge.py --rows 1000 --delay 0.01

# Or turbo stress-test load testing:
python boost.py --rows 5000
```

---

## 5. Technical Note on Feature Engineering

The original dataset includes a `Time` column indicating seconds elapsed since the first transaction. Because `Time` is strictly monotonically increasing, feeding it into an online streaming `StandardScaler` causes catastrophic statistical distortion during repeated stream passes. Therefore, the `Time` feature is explicitly dropped across all ingestion pipelines, preserving strictly translation-invariant features ($V_1$ through $V_{28}$ and `Amount`).
