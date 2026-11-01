# Payment Edge & Streaming Broker Fraud Detection Engine

A high-throughput, sub-millisecond **Online Machine Learning (Streaming ML)** system designed for **Payment Edge Gateways** and **Cloud Message Broker Architectures (Apache Kafka / Apache Flink)**.

---

## 1. System Positioning & Core Thesis

Traditional batch-learning models (e.g., XGBoost or Random Forest retrained on daily/weekly schedules) face fundamental challenges in high-velocity financial payment environments:
1. **Blind Window & Concept Drift Lag:** Emerging fraud patterns easily bypass static models during the multi-hour or multi-day lag between batch retraining cycles.
2. **Flooding by Legitimate Transactions & Catastrophic Forgetting:** In production payment networks where fraud accounts for under 0.2% of traffic, streaming models continuously updated only on clean transactions suffer from severe knowledge decay regarding historical fraud signatures.
3. **Inference Latency & Edge Footprints:** Bulky ensemble trees require significant compute and memory, making in-line sub-millisecond authorization impractical at distributed network edges.

This engine repositions real-time fraud detection into **two complementary production deployment topologies**:

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
  |  - Topic payment-transactions: High-throughput ingestion & distributed inference               |
  |  - Topic fraud-chargebacks: Delayed ground-truth feedback streaming for live weight updates    |
  +------------------------------------------------------------------------------------------------+
                                 |                                   |
                                 v                                   v
                    +---------------------------+       +---------------------------+
                    | Live Streamlit Monitoring |       | Cold Lake / Audit Storage |
                    | (PR-AUC, Pr@k, Drift, PER)|       | (Compliance & Forensics)  |
                    +---------------------------+       +---------------------------+
```

---

## 2. Multi-Model Architecture & Algorithmic Families

The system orchestrates and benchmarks five distinct model architectures across three machine learning paradigms:

| Model Architecture | Machine Learning Family | Update Mechanism | Primary Advantage | Operational Role |
| :--- | :--- | :--- | :--- | :--- |
| **Static Random Forest** | Offline Batch Learning (Scikit-Learn) | Static ($N=50$ trees, pre-trained on Prefix) | Stable baseline, immune to stream noise | Baseline Benchmark Control |
| **Logistic Regression (SGD)** | Online Linear Model (River) | Stochastic Gradient Descent (`learn_one`) | Extremely lightweight, $O(1)$ memory | Edge Gateway Line-Rate |
| **Logistic Regression (Adam)** | Online Linear Model (River) | Adaptive Moment Estimation ($m_t, v_t$) | Smooth convergence under steady distributions | Edge Gateway Line-Rate |
| **Hoeffding Adaptive Tree (HAT)** | Incremental Adaptive Decision Tree (River) | Hoeffding Bound split + ADWIN drift detector | Sub-tree replacement upon detecting concept drift | Streaming Non-linear Classifier |
| **Adaptive Random Forest (ARF)** | Online Ensemble Learning (River) | Online Bagging ($\lambda=6$) + ADWIN drift detector | Background tree replacement without interrupting inference | High-Accuracy Centralized Engine |

---

## 3. Advanced Mitigation of Catastrophic Forgetting & Drift Adaptation

To address extreme class imbalance (0.17% fraud rate) and continuous negative stream flooding, the engine implements two advanced online mechanisms:

### 3.1. Prioritized Experience Replay (PER) for Fraud Signatures
- A dedicated ring buffer maintains the most recent 100 verified fraud transactions.
- During high-volume legitimate transaction streams, the engine periodically triggers replay cycles every $N$ normal transactions (configurable, default $N=10$).
- Sampling priority is proportional to the prediction error:
  $$\delta_i = 1 - P(\text{fraud} \mid x_i)$$
  Hard positive samples (cases where the model assigned low fraud probability) receive the highest probability of replay, actively reinforcing decision boundaries.

### 3.2. Gaussian Noise Augmentation
- When replaying historical fraud samples, the engine injects Gaussian perturbations across numerical features:
  $$x' = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$
- Default perturbation intensity: $\sigma = 0.02$.
- This prevents the online trees from over-specializing on identical historical points and synthesizes nearby feature variations, expanding the effective generalization capacity against mutated fraud schemes.

### 3.3. Dual-Layer ADWIN Concept Drift Monitoring
- **River ADWIN (Adaptive Windowing):** Tracks streaming prediction error rates with dynamic statistical cutoffs ($\delta = 0.002$).
- When statistical drift is flagged, Hoeffding Trees begin growing alternative branches in parallel, swapping them in when their accuracy surpasses stale branches.

---

## 4. Experimental Protocol & Information Parity

To eliminate lookahead bias and data leakage, the ULB Credit Card Fraud dataset is split chronologically into two 50% partitions:

- **`data/creditcard_prefix.csv` (142,403 transactions, 269 frauds - 0.189% rate):** Historical warmup partition utilized exclusively by `pretrain.py` to overcome the cold-start penalty.
- **`data/creditcard_suffix.csv` (142,404 transactions, 223 frauds - 0.157% rate):** Strictly locked streaming evaluation dataset consumed in temporal sequence by stream producers (`payment_edge.py`, `boost.py`).

### Scientific Evaluation Metrics
Under extreme imbalance, standard ROC-AUC and Accuracy produce misleadingly inflated scores. This system enforces strict operational metrics:
- **Rolling PR-AUC (`RollingPRAUC`):** Prequential Area Under the Precision-Recall curve calculated via self-balancing binary search tree over a sliding window ($W=500$ to $1000$).
- **$Pr@k$ (Precision at $k$):** Precision within the top-$k$ riskiest transactions matching bank investigation capacity:
  $$k = \max(1, \text{round}(\alpha \times \text{fraud\_rate} \times W))$$
  where $\alpha = 1.0$ (default alert budget multiplier) and $W$ is the sliding window size.

---

## 5. Empirical Benchmark Results (30,000+ Transactions)

In a sustained stress test over 30,000 streaming transactions with injected concept drift:

| Model Architecture | Fraud Detections (True Positives) | Total Ground-Truth Frauds | Global Recognition Rate | Adaptation Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **Hoeffding Adaptive Tree (HAT)** | **109** | 117 | **93.2%** | Rapid branch pruning via ADWIN, high agility |
| **Adaptive Random Forest (ARF - Optimize)** | **98** | 117 | **83.8%** | Robust ensemble stability, enhanced via PER and Noise Augmentation |
| **Static Batch Random Forest** | **91** | 117 | **77.8%** | Significant performance degradation during concept drift |

---

## 6. Project Structure

```
OnlineML-FraudDetection/
|-- data/
|   |-- creditcard_prefix.csv         # 50% chronological warm-up data (142,403 rows)
|   +-- creditcard_suffix.csv         # 50% locked streaming evaluation data (142,404 rows)
|-- docs/
|   |-- Bao_Cao_Thu_Hoach_OnlineML_BKTPHCM.docx   # Comprehensive academic technical report
|   |-- accuracy_30k.png              # Empirical accuracy dashboard verification
|   +-- build_full_report_docx.py     # Formal Word report generation script
|-- boost.py                          # Turbo streaming generator with optional Concept Drift
|-- payment_edge.py                   # Paced Payment Edge Gateway simulator
|-- pretrain.py                       # Pre-training and serialization of baseline models
|-- server.py                         # High-performance FastAPI streaming inference server
|-- split_data.py                     # Chronological dataset partitioning utility
|-- view.py                           # Clean standard operational Streamlit dashboard
|-- view_optimize.py                  # Full deep-dive Optimize dashboard (PER & Augmentation)
|-- requirements.txt                  # Python dependencies
+-- README.md                         # Project documentation and specifications
```

---

## 7. Getting Started

### 7.1. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/Lomgz/OnlineML-FraudDetection.git
cd OnlineML-FraudDetection

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 7.2. Data Preparation & Pre-training
If `data/creditcard_prefix.csv` and `data/creditcard_suffix.csv` are not present, partition the dataset and pre-train the models:
```powershell
# 1. Chronological split of creditcard.csv (if raw CSV is present)
python split_data.py

# 2. Warm up and serialize initial model weights
python pretrain.py
```
*Outputs: `model_hat.pkl`, `model_arf.pkl`, `model_rf.pkl`, `model_sgd.pkl`, `model_adam.pkl`.*

### 7.3. Running the Streaming System

Open three separate terminal windows with virtual environment activated:

**Terminal 1: Start the Streaming ML Server**
```powershell
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2: Launch the Monitoring Dashboard**
```powershell
# Option A: Standard operational monitor
streamlit run view.py

# Option B: Advanced Optimize monitor (PER, Noise Augmentation, Hard Positives)
streamlit run view_optimize.py
```

**Terminal 3: Stream Transactions to the Server**
```powershell
# Standard edge streaming with 10ms pacing:
python payment_edge.py --rows 1000 --delay 0.01

# Or high-throughput turbo load testing:
python boost.py --rows 5000

# Or stress-testing with Concept Drift:
python boost.py --rows 10000 --drift --drift-start 1500 --drift-len 800
```

---

## 8. REST API Reference

The FastAPI streaming engine exposes the following production endpoints:

- `POST /predict`: Real-time transaction scoring. Returns individual probabilities across all models and gateway action (`ALLOW` vs `BLOCK` based on configurable threshold, default `0.70`).
- `POST /feedback`: Asynchronous ground-truth label ingestion. Executes `learn_one()`, records prediction error, triggers ADWIN drift checks, and updates the PER buffer.
- `GET /stats`: Returns live streaming metrics (Rolling PR-AUC, Precision@k, cumulative confusion matrices, global recognition counts).
- `GET /drift_info`: Returns status of ADWIN drift detector, warning flags, and historical drift timestamps.
- `GET /replay_info`: Returns current PER buffer occupancy, total replays executed, top hardest fraud samples, and original vs augmented feature comparisons.
- `POST /replay_config`: Dynamically updates PER and Gaussian noise hyperparameters (`enable_per`, `enable_noise_aug`, `noise_std`, `replay_interval`).
- `POST /config`: Configures operational parameters in real time (`block_threshold`, `alert_budget_alpha`, `window_size`).
- `GET /logs`: Serves the in-memory circular audit buffer (latest transactions with probability traces and decisions).

---

## 9. Feature Processing Specification

The raw ULB dataset contains a `Time` column denoting elapsed seconds from the initial recording. Because `Time` increases monotonically, feeding it into a streaming `StandardScaler` induces continuous statistical distortion during stream replay. Therefore, the `Time` feature is explicitly dropped across all ingestion pipelines, preserving strictly translation-invariant features ($V_1$ through $V_{28}$ and `Amount`).
