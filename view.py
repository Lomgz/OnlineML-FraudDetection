import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

def fetch_pr_curve(limit: int = 500) -> dict:
    try:
        r = requests.get(f"{SERVER}/pr_curve?limit={limit}", timeout=2)
        return r.json()
    except Exception:
        return None

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

    st.divider()
    st.subheader("Thong Tin Kien Truc")
    st.markdown("""
- **Dinh vi:** Payment Edge Gateways
- **Mo hinh:** Online Logistic Regression (SGD & Adam)
- **Chuan danh gia:** Experimental Protocol v2
- **Du lieu:** ULB Suffix 50% (142,404 txs)
    """)

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
    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
    col_kpi1.metric("Tong Giao Dich", f"{stats.get('total', 0):,}")
    col_kpi2.metric("Gian Lan Phat Hien", stats.get("fraud_detected", 0), delta=None, delta_color="inverse")
    col_kpi3.metric("So Lenh Bi Chan", stats.get("blocked", 0))

    prauc_sgd_val = stats.get('prauc_sgd', 0.0)
    sk_ap_sgd_val = stats.get('sk_ap_sgd', 0.0)
    prauc_adam_val = stats.get('prauc_adam', 0.0)
    sk_ap_adam_val = stats.get('sk_ap_adam', 0.0)

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

    st.divider()

    # Tab Navigation
    tab_monitor, tab_pr_analytics = st.tabs([
        "Giam Sat Luong Giao Dich (Streaming Monitor)",
        "Phan Tich Duong Cong Precision-Recall (Protocol v2)"
    ])

    # ---------------------------------------------------------
    # TAB 1: Streaming Monitor
    # ---------------------------------------------------------
    with tab_monitor:
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
                    y=["prob_sgd", "prob_adam"],
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
            st.subheader("Hieu Nang Mo Hinh Theo Chuan Protocol v2")
            col_perf1, col_perf2 = st.columns(2)

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

            st.divider()
            st.subheader("Nhat Ky Giao Dich Gan Nhat (Audit Log)")
            display_df = df[["transaction_id", "prob_sgd", "prob_adam", "action", "timestamp"]].copy()
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
                    "action": st.column_config.TextColumn("Quyet Dinh"),
                }
            )
        else:
            st.info("Chua co du lieu giao dich. Vui long chay lenh python payment_edge.py de bat dau gui luong thanh toan.")

    # ---------------------------------------------------------
    # TAB 2: Precision-Recall Curve & Analytics
    # ---------------------------------------------------------
    with tab_pr_analytics:
        st.subheader("Duong Cong Precision-Recall Thuc Te (Scikit-Learn)")
        st.caption("Truc quan hoa su danh doi giua Precision va Recall tren cua so giao dich hoat dong theo thoi gian thuc.")

        pr_data = fetch_pr_curve(limit=window_size)

        if pr_data and pr_data.get("frauds_in_window", 0) > 0:
            sgd_p = pr_data["sgd"]["precision"]
            sgd_r = pr_data["sgd"]["recall"]
            adam_p = pr_data["adam"]["precision"]
            adam_r = pr_data["adam"]["recall"]

            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=sgd_r, y=sgd_p, mode="lines+markers",
                name=f"SGD (Sklearn AP = {sk_ap_sgd_val:.4f})",
                line=dict(color="#3b82f6", width=2.5)
            ))
            fig_pr.add_trace(go.Scatter(
                x=adam_r, y=adam_p, mode="lines+markers",
                name=f"Adam (Sklearn AP = {sk_ap_adam_val:.4f})",
                line=dict(color="#10b981", width=2.5, dash="dash")
            ))
            fig_pr.update_layout(
                title=f"Precision-Recall Curve (Cua so: {pr_data.get('window_size', 0)} giao dich, {pr_data.get('frauds_in_window', 0)} ca gian lan)",
                xaxis_title="Recall (Do nhay bat gian lan)",
                yaxis_title="Precision (Do chuan xac canh bao)",
                template="plotly_dark",
                height=420,
                xaxis=dict(range=[0, 1.05]),
                yaxis=dict(range=[0, 1.05]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                uirevision="constant_fixed_view",
            )
            st.plotly_chart(fig_pr, use_container_width=True, key="chart_pr_curve")
        else:
            st.warning(f"Cua so hien tai ({stats.get('window_size', 0)} giao dich) chua co ca gian lan nao (True Fraud = 0) de ve duong cong Precision-Recall day du. Hay tiep tuc stream giao dich tu payment_edge.py.")

        st.divider()
        st.subheader("Ra Soat & Doi Chieu Phuong Phap Tinh PR-AUC")
        st.markdown("""
| Thuoc Do | Co Che Toan Hoc | Vai Tro Trong Du An | Uu / Nhuoc Diem |
| :--- | :--- | :--- | :--- |
| **River `RollingPRAUC`** | Cay nhi phan tu can bang (Gomes et al., 2023) tinh dien tich Prequential AUC-PR tren stream $O(\\log W)$. | Đo luong hieu nang truc tuyen lien tuc tai Edge Gateway. | **Uu diem:** Toc do sieu nhanh, bo nho $O(W)$, cap nhat tung mau.<br>**Luu y:** Voi cua so chi co 1 fraud don le, dien tich co the tiem can 1.0. |
| **Scikit-Learn `Average Precision (AP)`** | Tinh dien tich theo phuong phap noi suy trung binh trong so: $AP = \\sum_n (R_n - R_{n-1}) P_n$. | Doi chieu chuan muc khoa hoc (Gold Standard) tren cua so giao dich. | **Uu diem:** Khach quan tuyet doi, khong bi phong dai boi quy tac hinh thang.<br>**Luu y:** Chi xac dinh khi cua so co $\\ge 1$ mau gian lan. |
| **Chi so $Pr@k$ (Protocol v2)** | Ty le gian lan that trong top $k$ giao dich co diem rui ro cao nhat: $k = \\max(1, \\text{round}(\\alpha \\times \\text{rate} \\times W))$. | Thuoc do quyet dinh gia thuyet nghien cuu $H_1$ theo ngan sach dieu tra cua ngan hang. | **Uu diem:** Sat thuc te van hanh nhat: giup doi phong chong gian lan biet xem xet $k$ canh bao thi bat trung bao nhieu ca that. |
        """)

# Render fragment
render_dashboard()
