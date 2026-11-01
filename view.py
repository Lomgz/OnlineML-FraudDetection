import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import time

st.set_page_config(
    page_title="Payment Edge Fraud Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Disable Streamlit dimming / gray-out effect during refresh & fragment execution
NO_DIM_CSS = """
<style>
/* Triệt tiêu 100% hiệu ứng làm mờ / xám khi Streamlit hoặc Fragment refresh */
html, body, [class*="css"], [class*="st-"], .stApp,
.stApp *,
div[data-testid="stAppViewContainer"],
div[data-testid="stAppViewContainer"] *,
section.main,
section.main *,
div.block-container,
div.block-container *,
[data-testid="stFragment"],
[data-testid="stFragment"] *,
[data-st-fragment-id],
[data-st-fragment-id] *,
[data-test-fragment-state="running"],
[data-test-fragment-state="running"] *,
.stApp[data-test-script-state="running"],
.stApp[data-test-script-state="running"] *,
div[data-testid="stVerticalBlock"],
div[data-testid="stVerticalBlock"] *,
div[data-testid="stElementContainer"],
div[data-testid="stElementContainer"] *,
div[data-testid="stPlotlyChart"],
div[data-testid="stPlotlyChart"] * {
    opacity: 1 !important;
    filter: none !important;
    transition: none !important;
    animation: none !important;
}

/* Ẩn hoàn toàn spinner widget báo hiệu đang chạy ở góc trên bên phải */
[data-testid="stStatusWidget"],
div[data-testid="stStatusWidget"],
div[data-testid="stDecoration"] {
    display: none !important;
    visibility: hidden !important;
}

/* Tắt chuyển động làm chậm của Plotly */
.js-plotly-plot,
.js-plotly-plot .plotly,
.js-plotly-plot .plotly .main-svg {
    transition: none !important;
    opacity: 1 !important;
}
</style>
"""
st.markdown(NO_DIM_CSS, unsafe_allow_html=True)

SERVER = "http://localhost:8000"

def fetch_stats() -> dict:
    try:
        r = requests.get(f"{SERVER}/stats", timeout=2)
        return r.json()
    except Exception:
        return None

def fetch_logs(limit: int = 100) -> list:
    try:
        r = requests.get(f"{SERVER}/logs?limit={limit}", timeout=2)
        return r.json().get("transactions", [])
    except Exception:
        return []


def update_server_config(threshold: float, alpha: float, window_size: int):
    try:
        requests.post(
            f"{SERVER}/config",
            json={
                "block_threshold": threshold,
                "alert_budget_alpha": alpha,
                "window_size": window_size,
            },
            timeout=2
        )
    except Exception:
        pass

# -------------------------------------------------------------
# SIDEBAR: Cau hinh Cong Thanh toan Bien
# -------------------------------------------------------------
with st.sidebar:
    st.header("Cau Hinh Cong Bien")
    st.caption("Payment Edge Gateway & ML Engine Settings")

    current_stats = fetch_stats()
    srv_config = current_stats.get("config", {}) if current_stats else {}
    initial_thresh = float(srv_config.get("block_threshold", 0.70))
    initial_alpha = float(srv_config.get("alert_budget_alpha", 1.0))
    initial_win = int(srv_config.get("window_size", 500))

    threshold = st.slider(
        "Nguong chan (Block Threshold)",
        min_value=0.05,
        max_value=0.95,
        value=initial_thresh,
        step=0.05,
        help="Giao dich co P(fraud) > nguong nay se bi Cong thanh toan chan (BLOCK) tuc thi."
    )

    alpha = st.slider(
        "He so Alert Budget (alpha)",
        min_value=0.1,
        max_value=3.0,
        value=initial_alpha,
        step=0.1,
        help="He so alpha dung tinh k trong Pr@k: k = max(1, round(alpha * fraud_rate * window_size))."
    )

    window_size = st.slider(
        "Cua so truot (Window Size W)",
        min_value=100,
        max_value=2000,
        value=initial_win,
        step=100,
        help="So luong giao dich gan nhat dung de tinh toan Pr@k, AP va PR Curve."
    )

    refresh_interval = st.slider(
        "Chu ky lam moi (Refresh Sec)",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help="Thoi gian tu dong lam moi giao dien dashboard (giay)."
    )

    if (threshold != initial_thresh or alpha != initial_alpha or window_size != initial_win):
        update_server_config(threshold, alpha, window_size)

# -------------------------------------------------------------
# MAIN VIEW: Non-flickering Auto-refreshing Fragment
# -------------------------------------------------------------
st.title("Payment Edge Fraud Detection Engine")
st.caption(f"Streaming Engine URL: `{SERVER}` | Block Threshold: `{threshold:.2f}` | Alert Budget alpha: `{alpha:.1f}` | Refresh: `{refresh_interval}s`")

@st.fragment(run_every=refresh_interval)
def render_dashboard():
    # Inject anti-dimming styles inside fragment to override any Streamlit runtime transitions
    st.markdown(NO_DIM_CSS, unsafe_allow_html=True)
    stats = fetch_stats()

    if stats is None:
        st.error("May chu ML Server dang Offline! Vui long dam bao server.py dang chay tai cong 8000.")
        return

    st.success("ML Server Online - San sang tiep nhan luong giao dich bien")

    # Top KPI Metric Cards
    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5, col_kpi6 = st.columns(6)
    col_kpi1.metric("Tong Giao Dich", f"{stats.get('total', 0):,}")
    col_kpi2.metric("Gian Lan Phat Hien", stats.get("fraud_detected", 0), delta=None, delta_color="inverse")
    col_kpi3.metric("So Lenh Bi Chan", stats.get("blocked", 0))

    prauc_sgd_val = stats.get('prauc_sgd', 0.0)
    sk_ap_sgd_val = stats.get('sk_ap_sgd', 0.0)
    prauc_adam_val = stats.get('prauc_adam', 0.0)
    sk_ap_adam_val = stats.get('sk_ap_adam', 0.0)
    prauc_rf_val = stats.get('prauc_rf', 0.0) 
    sk_ap_rf_val = stats.get('sk_ap_rf', 0.0)
    pr_k_rf_pct = stats.get('pr_at_k_rf', 0.0) * 100
    pr_k_sgd_pct = stats.get('pr_at_k_sgd', 0.0) * 100  
    pr_k_adam_pct = stats.get('pr_at_k_adam', 0.0) * 100
    
    col_kpi4.metric(
        "PR-AUC (SGD)",
        f"{prauc_sgd_val:.4f}",
        help=f"River RollingPRAUC: {prauc_sgd_val:.4f} | Sklearn AP: {sk_ap_sgd_val:.4f}"
    )
    col_kpi5.metric(
        "PR-AUC (Adam)",
        f"{prauc_adam_val:.4f}",
        help=f"River RollingPRAUC: {prauc_adam_val:.4f} | Sklearn AP: {sk_ap_adam_val:.4f}"
    )
    col_kpi6.metric(
        "PR-AUC (RF)",
        f"{prauc_rf_val:.4f}",
        help=f"River RollingPRAUC: {prauc_rf_val:.4f} | Sklearn AP: {sk_ap_rf_val:.4f}"
    )
    st.divider()


    logs = fetch_logs(limit=100)
    if logs:
        df = pd.DataFrame(logs)
        df["index"] = range(len(df))
        df["color"] = df["action"].map({"BLOCK": "#ef4444", "ALLOW": "#22c55e"})

        col_chart1, col_chart2 = st.columns([2, 1])

        with col_chart1:
            st.subheader("Xac Suat Gian Lan Theo Thoi Gian")
            fig_line = px.line(
                df,
                x="index",
                y=["prob_sgd", "prob_adam", "prob_rf"], 
                labels={"index": "Thu tu Giao dich", "value": "P(fraud)", "variable": "Mo hinh"},
                template="plotly_dark",
                height=320,
            )
            fig_line.add_hline(
                y=threshold, line_dash="dash", line_color="#facc15",
                annotation_text=f"Nguong Chan {threshold:.2f}"
            )
            fig_line.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                uirevision="constant_fixed_view",
            )
            st.plotly_chart(fig_line, use_container_width=True, key="chart_line_stream")

        with col_chart2:
            st.subheader("Ty Le ALLOW / BLOCK")
            action_counts = df["action"].value_counts().reset_index()
            action_counts.columns = ["action", "count"]
            fig_pie = px.pie(
                action_counts,
                names="action",
                values="count",
                color="action",
                color_discrete_map={"BLOCK": "#ef4444", "ALLOW": "#22c55e"},
                template="plotly_dark",
                height=320,
                hole=0.5,
            )
            fig_pie.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                uirevision="constant_fixed_view",
            )
            st.plotly_chart(fig_pie, use_container_width=True, key="chart_pie_action")

        st.divider()
        st.subheader("Hieu Nang Mo Hinh (Batch vs Online)")
        st.markdown("### Bảng Tổng Hợp Benchmark (Batch vs Online)")
        benchmark_data = {
            "Model": ["Random Forest (Sklearn)", "Logistic Regression (SGD - River)", "Logistic Regression (Adam - River)"],
            "Family": ["Batch (Offline Baseline)", "Online (Streaming)", "Online (Streaming)"],
            "Mean PR-AUC": [f"{prauc_rf_val:.4f}", f"{prauc_sgd_val:.4f}", f"{prauc_adam_val:.4f}"],
            "Precision@k": [f"{stats.get('pr_at_k_rf', 0.0)*100:.2f}%", f"{stats.get('pr_at_k_sgd', 0.0)*100:.2f}%", f"{stats.get('pr_at_k_adam', 0.0)*100:.2f}%"]
        }
        st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True)
        st.divider()

        col_perf1, col_perf2, col_perf3 = st.columns(3)

        k_val = stats.get("k_eval", 1)
        win_size = stats.get("window_size", window_size)
        frauds_in_win = stats.get("frauds_in_window", 0)

        with col_perf1:
            st.markdown(f"""
**SGD Pr@{k_val} (Top-{k_val} Alert Budget)**: `{stats.get('pr_at_k_sgd', 0.0)*100:.2f}%`
* **River Rolling PR-AUC**: `{prauc_sgd_val:.4f}`
* **Scikit-learn Average Precision (AP)**: `{sk_ap_sgd_val:.4f}`
* **Cua so tinh toan**: `{win_size}` giao dich gan nhat (`{frauds_in_win}` ca gian lan)
```text
Ma tran nham lan tich luy (SGD - {stats.get('total', 0):,} txs):
True Positives : {stats.get('cm_sgd', {}).get('tp', 0):<6} | False Positives: {stats.get('cm_sgd', {}).get('fp', 0):<6}
False Negatives: {stats.get('cm_sgd', {}).get('fn', 0):<6} | True Negatives : {stats.get('cm_sgd', {}).get('tn', 0):<6}
```
""")

        with col_perf2:
            st.markdown(f"""
**Adam Pr@{k_val} (Top-{k_val} Alert Budget)**: `{stats.get('pr_at_k_adam', 0.0)*100:.2f}%`
* **River Rolling PR-AUC**: `{prauc_adam_val:.4f}`
* **Scikit-learn Average Precision (AP)**: `{sk_ap_adam_val:.4f}`
* **Cua so tinh toan**: `{win_size}` giao dich gan nhat (`{frauds_in_win}` ca gian lan)
```text
Ma tran nham lan tich luy (Adam - {stats.get('total', 0):,} txs):
True Positives : {stats.get('cm_adam', {}).get('tp', 0):<6} | False Positives: {stats.get('cm_adam', {}).get('fp', 0):<6}
False Negatives: {stats.get('cm_adam', {}).get('fn', 0):<6} | True Negatives : {stats.get('cm_adam', {}).get('tn', 0):<6}
```
""")
        with col_perf3:
            st.markdown(f"""
**Batch RF Pr@{k_val} (Top-{k_val} Alert Budget)**: `{pr_k_rf_pct:.2f}%`
* **River Rolling PR-AUC**: `{prauc_rf_val:.4f}`
* **Scikit-learn Average Precision (AP)**: `{sk_ap_rf_val:.4f}`
* **Cua so tinh toan**: `{win_size}` giao dich gan nhat (`{frauds_in_win}` ca gian lan)
```text
Ma tran nham lan tich luy (Batch RF - {stats.get('total', 0):,} txs):
True Positives : {stats.get('cm_rf', {}).get('tp', 0):<6} | False Positives: {stats.get('cm_rf', {}).get('fp', 0):<6}
False Negatives: {stats.get('cm_rf', {}).get('fn', 0):<6} | True Negatives : {stats.get('cm_rf', {}).get('tn', 0):<6}
""")   
        st.divider()
        st.subheader("Nhat Ky Giao Dich Gan Nhat (Audit Log)")
        display_df = df[["transaction_id", "prob_sgd", "prob_adam", "prob_rf", "action", "timestamp"]].copy()
        display_df["timestamp"] = pd.to_datetime(display_df["timestamp"], unit="s").dt.strftime("%H:%M:%S")
        display_df = display_df.iloc[::-1].reset_index(drop=True)
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "prob_sgd": st.column_config.ProgressColumn(
                    "P(fraud) SGD", min_value=0, max_value=1, format="%.4f"
                ),
                "prob_adam": st.column_config.ProgressColumn(
                    "P(fraud) Adam", min_value=0, max_value=1, format="%.4f"
                ),
                "prob_rf": st.column_config.ProgressColumn(
                    "P(fraud) RF", min_value=0, max_value=1, format="%.4f"
),
                "action": st.column_config.TextColumn("Quyet Dinh"),
            }
        )
    else:
        st.info("Chua co du lieu giao dich. Vui long chay lenh python payment_edge.py de bat dau gui luong thanh toan.")


# Render fragment
render_dashboard()
