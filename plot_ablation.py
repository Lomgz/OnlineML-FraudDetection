import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUMMARY_CSV = "results/ablation_summary.csv"
CURVE_CSV = "results/ablation_curve.csv"
OUT_DIR = "results"

DRIFT_START = 10000
DRIFT_END = 13000

MODEL_LABELS = {
    "M0_Batch_RF": "M0: Batch RF",
    "M1_ARF_Pure": "M1: ARF Pure",
    "M2_ARF_Weight250": "M2: +Weight250",
    "M3_ARF_UniformReplay": "M3: +Uniform",
    "M4_ARF_PER": "M4: +PER",
    "M5_ARF_PER_Noise": "M5: +PER+Noise",
    "M6_ARF_Full_ADWIN": "M6: +ADWIN",
}

MODEL_COLORS = {
    "M0_Batch_RF": "#64748b",
    "M1_ARF_Pure": "#94a3b8",
    "M2_ARF_Weight250": "#ef4444",
    "M3_ARF_UniformReplay": "#f59e0b",
    "M4_ARF_PER": "#3b82f6",
    "M5_ARF_PER_Noise": "#10b981",
    "M6_ARF_Full_ADWIN": "#8b5cf6",
}


def plot_drift_recovery(curve_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=180)

    # Vung Concept Drift
    ax.axvspan(DRIFT_START, DRIFT_END, color="#fecaca", alpha=0.35, label="Concept Drift (10,000 - 13,000)")

    plot_models = [
        ("M0_Batch_RF", ":", 1.8),
        ("M1_ARF_Pure", "--", 1.8),
        ("M2_ARF_Weight250", "-.", 1.8),
        ("M4_ARF_PER", "-", 2.2),
        ("M5_ARF_PER_Noise", "-", 2.4),
        ("M6_ARF_Full_ADWIN", "-", 2.2),
    ]

    for mid, ls, lw in plot_models:
        sub = curve_df[curve_df["model_id"] == mid].sort_values("step")
        if sub.empty:
            continue
        ax.plot(
            sub["step"],
            sub["rolling_prauc"],
            label=MODEL_LABELS.get(mid, mid),
            color=MODEL_COLORS.get(mid, "#333333"),
            linestyle=ls,
            linewidth=lw,
            alpha=0.92,
        )

    # Danh dau cac diem ADWIN phat hien Drift o M6
    m6_drifts = curve_df[(curve_df["model_id"] == "M6_ARF_Full_ADWIN") & (curve_df["drift_detected"] == 1)]
    if not m6_drifts.empty:
        ax.scatter(
            m6_drifts["step"],
            m6_drifts["rolling_prauc"],
            color="#facc15",
            edgecolor="#1e293b",
            marker="D",
            s=45,
            zorder=6,
            label=f"M6 ADWIN Drift Trigger ({len(m6_drifts)} lần)",
        )

    ax.set_title("Biến thiên Rolling PR-AUC (W=1000) trên luồng 30.000 giao dịch có Delayed Feedback & Drift", fontsize=12, fontweight="bold", pad=42)
    ax.set_xlabel("Số giao dịch đã nhận nhãn phản hồi (Labeled Step)", fontsize=10.5)
    ax.set_ylabel("Rolling PR-AUC", fontsize=10.5)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, fontsize=8.5, frameon=True)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig1_prauc_drift_recovery.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"[Plot] Da luu: {out_path}")


def plot_metrics_comparison(sum_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=180)

    labels = [MODEL_LABELS.get(m, m) for m in sum_df["model_id"]]
    x = np.arange(len(labels))
    width = 0.2

    metrics_cfg = [
        ("overall_ap", "Overall AP", "#0ea5e9"),
        ("mean_rolling_prauc", "Mean Rolling PR-AUC", "#10b981"),
        ("mean_pr_at_k", "Mean Pr@k", "#8b5cf6"),
        ("f2", "F2-Score", "#f59e0b"),
    ]

    for i, (col, name, color) in enumerate(metrics_cfg):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, sum_df[col], width, label=name, color=color, edgecolor="#1e293b", linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_title("So sánh hiệu năng Ablation Study (M0 - M6) trên 30.000 giao dịch", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, rotation=12)
    ax.set_ylabel("Giá trị thước đo (0.0 - 1.0)", fontsize=10.5)
    ax.set_ylim(0, 0.75)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", ncol=4, fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig2_ablation_metrics.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"[Plot] Da luu: {out_path}")


def plot_tradeoff_and_sla(sum_df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=180)
    labels = [MODEL_LABELS.get(m, m) for m in sum_df["model_id"]]
    colors = [MODEL_COLORS.get(m, "#333333") for m in sum_df["model_id"]]

    # Panel trai: Danh doi Precision vs False Positives
    bars_fp = ax1.bar(labels, sum_df["fp"], color=colors, edgecolor="#1e293b", linewidth=0.6)
    ax1.set_title("Đánh đổi Cảnh báo giả (False Positives) & Precision", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Số lượng False Positives (FP)", fontsize=10)
    ax1.tick_params(axis="x", rotation=25, labelsize=8.5)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)

    for bar, prec in zip(bars_fp, sum_df["precision"]):
        h = bar.get_height()
        ax1.annotate(
            f"FP={int(h)}\n({prec*100:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax1.set_ylim(0, max(sum_df["fp"]) * 1.25 + 2)

    # Panel phai: Do tre suy luan so voi SLA < 5ms
    bars_lat = ax2.bar(labels, sum_df["avg_latency_ms"], color=colors, edgecolor="#1e293b", linewidth=0.6)
    ax2.axhline(5.0, color="#dc2626", linestyle="--", linewidth=1.8, label="Ngưỡng SLA (5.0 ms)")
    ax2.set_title("Độ trễ suy luận trung bình tại cổng biên (SLA < 5ms)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Độ trễ trung bình (ms / giao dịch)", fontsize=10)
    ax2.tick_params(axis="x", rotation=25, labelsize=8.5)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax2.legend(loc="upper right", fontsize=9)

    for bar in bars_lat:
        h = bar.get_height()
        ax2.annotate(
            f"{h:.2f} ms",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax2.set_ylim(0, max(sum_df["avg_latency_ms"]) * 1.2)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig3_tradeoff_fp_sla.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"[Plot] Da luu: {out_path}")


def main():
    if not os.path.exists(SUMMARY_CSV) or not os.path.exists(CURVE_CSV):
        print("[Error] Khong tim thay file CSV trong thu muc results/. Hay chay run_ablation.py truoc!")
        return

    sum_df = pd.read_csv(SUMMARY_CSV)
    curve_df = pd.read_csv(CURVE_CSV)

    plot_drift_recovery(curve_df)
    plot_metrics_comparison(sum_df)
    plot_tradeoff_and_sla(sum_df)
    print("[Done] Hoan tat xuat 3 bieu do trong thu muc results/.")


if __name__ == "__main__":
    main()