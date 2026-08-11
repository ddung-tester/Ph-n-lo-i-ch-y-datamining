"""
final_evaluation_official.py - Final Evaluation & Error Analysis cho BTL Data Mining.

MỤC ĐÍCH:
1. Tổng hợp kết quả từ model_comparison_OFFICIAL.csv và cross_validation_OFFICIAL.csv.
2. Vẽ các biểu đồ so sánh 4 mô hình (Accuracy, F1, Recall, Runtime, CV F1, CV Recall).
3. Vẽ Confusion Matrix cho cả 4 mô hình từ số liệu thực tế.
4. Phân tích lỗi (Error Analysis) chi tiết trên tập TEST cho 2 mô hình tốt nhất: SVM và Random Forest.
   - Trích xuất 5 mẫu False Positive (FP) và 5 mẫu False Negative (FN).
   - Lưu ảnh mẫu vào reports/errors/{svm,random_forest}/{false_positive,false_negative}/.
   - Tạo contact sheet minh họa: reports/figures/{svm,random_forest}_error_examples_OFFICIAL.png.
5. Xuất tệp tổng hợp cuối cùng: reports/results/final_summary_OFFICIAL.csv.

LƯU Ý: KHÔNG TRAIN LẠI BẤT KỲ MODEL NÀO. Tái sử dụng 100% tài nguyên đã huấn luyện.
"""

import os
import sys
import time
import csv
import shutil
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend hiển thị không cần GUI
import matplotlib.pyplot as plt
import seaborn as sns

# Reconfigure stdout cho Windows console để hiển thị UTF-8 không bị lỗi
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Đảm bảo PROJECT_ROOT trong sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import (
    RESULTS_DIR, FIGURES_DIR, MODELS_DIR, ensure_dir, setup_logger, imread_unicode
)

logger = setup_logger("final_evaluation_official")

# Đường dẫn tệp đầu vào
MODEL_COMP_CSV_PATH = os.path.join(RESULTS_DIR, "model_comparison_OFFICIAL.csv")
CV_COMP_CSV_PATH = os.path.join(RESULTS_DIR, "cross_validation_OFFICIAL.csv")
TEST_X_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "test", "X_combined.npy")
TEST_Y_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "test", "y.npy")
TEST_META_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "test", "metadata.csv")

OFFICIAL_MODELS_DIR = os.path.join(MODELS_DIR, "official")
SVM_MODEL_PATH = os.path.join(OFFICIAL_MODELS_DIR, "svm_combined.pkl")
RF_MODEL_PATH = os.path.join(OFFICIAL_MODELS_DIR, "random_forest_combined.pkl")

# Đường dẫn tệp đầu ra
FINAL_SUMMARY_CSV_PATH = os.path.join(RESULTS_DIR, "final_summary_OFFICIAL.csv")
ERRORS_DIR = os.path.join(PROJECT_ROOT, "reports", "errors")


def log_msg(msg):
    """In log trực tiếp ra terminal với timestamp và flush ngay lập tức."""
    timestamp = time.strftime("[%H:%M:%S]")
    print(f"{timestamp} {msg}", flush=True)


def load_official_csv_results():
    """Load model_comparison_OFFICIAL.csv và cross_validation_OFFICIAL.csv."""
    if not os.path.exists(MODEL_COMP_CSV_PATH):
        raise FileNotFoundError(f"Không tìm thấy file {MODEL_COMP_CSV_PATH}")
    if not os.path.exists(CV_COMP_CSV_PATH):
        raise FileNotFoundError(f"Không tìm thấy file {CV_COMP_CSV_PATH}")

    df_model = pd.read_csv(MODEL_COMP_CSV_PATH)
    df_cv = pd.read_csv(CV_COMP_CSV_PATH)

    return df_model, df_cv


def create_model_comparison_charts(df_model):
    """Phần 1: Tạo các biểu đồ so sánh 4 mô hình trên Hold-out TEST."""
    ensure_dir(FIGURES_DIR)

    models = df_model["model"].tolist()
    x = np.arange(len(models))
    width = 0.45
    colors = ["#42A5F5", "#AB47BC", "#FFA726", "#66BB6A"]

    # 1. Accuracy Comparison Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    accs = df_model["accuracy"].tolist()
    bars = ax.bar(x, accs, width, color=colors, edgecolor="black")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.01, f"{h:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=11, fontweight="bold")
    ax.set_title("Official Model Comparison - Accuracy (TEST Set)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    acc_fig_path = os.path.join(FIGURES_DIR, "model_accuracy_comparison_OFFICIAL.png")
    plt.savefig(acc_fig_path, dpi=150)
    plt.close()

    # 2. F1-score Comparison Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    f1s = df_model["f1_fire"].tolist()
    bars = ax.bar(x, f1s, width, color=colors, edgecolor="black")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.01, f"{h:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("F1 Score (Fire Class)", fontsize=11, fontweight="bold")
    ax.set_title("Official Model Comparison - F1 Score (Fire Class)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    f1_fig_path = os.path.join(FIGURES_DIR, "model_f1_comparison_OFFICIAL.png")
    plt.savefig(f1_fig_path, dpi=150)
    plt.close()

    # 3. Recall Comparison Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    recs = df_model["recall_fire"].tolist()
    bars = ax.bar(x, recs, width, color=colors, edgecolor="black")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.01, f"{h:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Recall Score (Fire Class)", fontsize=11, fontweight="bold")
    ax.set_title("Official Model Comparison - Recall Score (Fire Class)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    rec_fig_path = os.path.join(FIGURES_DIR, "model_recall_comparison_OFFICIAL.png")
    plt.savefig(rec_fig_path, dpi=150)
    plt.close()

    # 4. Runtime Comparison Chart (Train & Predict)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    train_times = df_model["train_time"].tolist()
    pred_times = df_model["prediction_time"].tolist()
    bar_w = 0.35
    b_tr = ax.bar(x - bar_w / 2, train_times, bar_w, label="Train Time (s)", color="#26A69A", edgecolor="black")
    b_pr = ax.bar(x + bar_w / 2, pred_times, bar_w, label="Predict Time (s)", color="#FF7043", edgecolor="black")

    max_val = max(max(train_times), max(pred_times))
    for bars in [b_tr, b_pr]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + (max_val * 0.01), f"{h:.1f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax.set_title("Official Model Comparison - Execution Time (Seconds)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(0, max_val * 1.18)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    runtime_fig_path = os.path.join(FIGURES_DIR, "model_runtime_comparison_OFFICIAL.png")
    plt.savefig(runtime_fig_path, dpi=150)
    plt.close()


def create_cross_validation_charts(df_cv):
    """Phần 2: Tạo các biểu đồ 5-Fold Cross-Validation (Mean ± Std)."""
    ensure_dir(FIGURES_DIR)

    models = df_cv["model"].tolist()
    x = np.arange(len(models))
    width = 0.45

    # 1. CV F1 Mean ± Std Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    f1_mean = df_cv["f1_fire_mean"].tolist()
    f1_std = df_cv["f1_fire_std"].tolist()

    bars = ax.bar(x, f1_mean, width, yerr=f1_std, capsize=5, color="#5C6BC0", edgecolor="black")
    for bar, m, s in zip(bars, f1_mean, f1_std):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + s + 0.015, f"{m:.4f}±{s:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("5-Fold CV F1-Score (Fire Class)", fontsize=11, fontweight="bold")
    ax.set_title("Official 5-Fold Cross-Validation - F1-Score (Mean ± Std)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    cv_f1_fig_path = os.path.join(FIGURES_DIR, "cv_f1_comparison_OFFICIAL.png")
    plt.savefig(cv_f1_fig_path, dpi=150)
    plt.close()

    # 2. CV Recall Mean ± Std Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    rec_mean = df_cv["recall_fire_mean"].tolist()
    rec_std = df_cv["recall_fire_std"].tolist()

    bars = ax.bar(x, rec_mean, width, yerr=rec_std, capsize=5, color="#EC407A", edgecolor="black")
    for bar, m, s in zip(bars, rec_mean, rec_std):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + s + 0.015, f"{m:.4f}±{s:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("5-Fold CV Recall Score (Fire Class)", fontsize=11, fontweight="bold")
    ax.set_title("Official 5-Fold Cross-Validation - Recall Score (Mean ± Std)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    cv_rec_fig_path = os.path.join(FIGURES_DIR, "cv_recall_comparison_OFFICIAL.png")
    plt.savefig(cv_rec_fig_path, dpi=150)
    plt.close()


def create_confusion_matrices(df_model):
    """Phần 3: Vẽ Confusion Matrix cho cả 4 mô hình dựa trên số liệu thực tế trong CSV."""
    ensure_dir(FIGURES_DIR)

    model_file_map = {
        "Logistic Regression": "cm_logistic_regression_OFFICIAL.png",
        "K-Nearest Neighbors": "cm_knn_OFFICIAL.png",
        "SVM": "cm_svm_OFFICIAL.png",
        "Random Forest": "cm_random_forest_OFFICIAL.png"
    }

    class_labels = ["non_fire (0)", "fire (1)"]

    for _, row in df_model.iterrows():
        model_name = row["model"]
        tp, tn, fp, fn = int(row["tp"]), int(row["tn"]), int(row["fp"]), int(row["fn"])

        cm = np.array([[tn, fp], [fn, tp]])
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

        # Heatmap số lượng
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=class_labels, yticklabels=class_labels, ax=axes[0],
                    annot_kws={"size": 14, "weight": "bold"})
        axes[0].set_title(f"{model_name} — Counts", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
        axes[0].set_ylabel("True Label", fontsize=10, fontweight="bold")

        # Heatmap tỷ lệ %
        sns.heatmap(cm_pct, annot=True, fmt=".2f", cmap="Oranges", cbar=False,
                    xticklabels=class_labels, yticklabels=class_labels, ax=axes[1],
                    annot_kws={"size": 14, "weight": "bold"})
        axes[1].set_title(f"{model_name} — Percentage (%)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
        axes[1].set_ylabel("True Label", fontsize=10, fontweight="bold")

        plt.suptitle(f"Official Confusion Matrix: {model_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        out_name = model_file_map.get(model_name, f"cm_{model_name.lower().replace(' ', '_')}_OFFICIAL.png")
        save_path = os.path.join(FIGURES_DIR, out_name)
        plt.savefig(save_path, dpi=150)
        plt.close()


def perform_error_analysis():
    """Phần 4: Error Analysis cho SVM và Random Forest trên tập TEST."""
    ensure_dir(ERRORS_DIR)

    log_msg("  Loading TEST dataset features & metadata...")
    X_test = np.load(TEST_X_PATH)
    y_test = np.load(TEST_Y_PATH)
    df_meta = pd.read_csv(TEST_META_PATH)

    models_to_analyze = [
        ("SVM", SVM_MODEL_PATH, "svm"),
        ("Random Forest", RF_MODEL_PATH, "random_forest")
    ]

    error_summary = {}

    for model_name, model_path, slug in models_to_analyze:
        log_msg(f"  [Error Analysis] Loading {model_name} model from {os.path.basename(model_path)}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file model {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        log_msg(f"  Predicting 6,122 TEST samples with {model_name}...")
        t0 = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - t0
        log_msg(f"  {model_name} prediction finished in {pred_time:.2f}s")

        # Kiểm tra xem model có hỗ trợ predict_proba không
        probs = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X_test)
            except Exception:
                probs = None

        # Tìm chỉ số FP và FN
        fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
        fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]

        log_msg(f"  {model_name} — Total FP (Non-fire -> Fire): {len(fp_indices)}, Total FN (Fire -> Non-fire): {len(fn_indices)}")

        error_summary[slug] = {
            "name": model_name,
            "fp_count": len(fp_indices),
            "fn_count": len(fn_indices),
            "fp_indices": fp_indices,
            "fn_indices": fn_indices
        }

        # Lưu ảnh mẫu (5 FP và 5 FN)
        fp_dir = os.path.join(ERRORS_DIR, slug, "false_positive")
        fn_dir = os.path.join(ERRORS_DIR, slug, "false_negative")
        ensure_dir(fp_dir)
        ensure_dir(fn_dir)

        # Chọn 5 mẫu đại diện
        selected_fp = fp_indices[:5]
        selected_fn = fn_indices[:5]

        # Copy tệp ảnh mẫu
        for idx in selected_fp:
            row = df_meta.iloc[idx]
            src_path = row["image_path"]
            if os.path.exists(src_path):
                shutil.copy2(src_path, os.path.join(fp_dir, os.path.basename(src_path)))

        for idx in selected_fn:
            row = df_meta.iloc[idx]
            src_path = row["image_path"]
            if os.path.exists(src_path):
                shutil.copy2(src_path, os.path.join(fn_dir, os.path.basename(src_path)))

        # Tạo contact sheet visualization (5 FP + 5 FN = 10 panels grid)
        fig, axes = plt.subplots(2, 5, figsize=(15, 6.5))
        axes = axes.flatten()

        sample_list = list(selected_fp) + list(selected_fn)

        for i, idx in enumerate(sample_list):
            ax = axes[i]
            row = df_meta.iloc[idx]
            src_path = row["image_path"]
            img_bgr = imread_unicode(src_path)

            if img_bgr is not None:
                img_rgb = img_bgr[:, :, ::-1]
                ax.imshow(img_rgb)
            else:
                ax.text(0.5, 0.5, "Image Not Found", ha="center", va="center")

            actual_label = "fire" if y_test[idx] == 1 else "non_fire"
            pred_label = "fire" if y_pred[idx] == 1 else "non_fire"

            prob_str = ""
            if probs is not None:
                prob_fire = probs[idx][1]
                prob_str = f"\nProb(fire): {prob_fire:.2f}"

            panel_title = f"{'FP' if i < 5 else 'FN'}: {row['filename']}\nActual: {actual_label} | Pred: {pred_label}{prob_str}"
            title_color = "red" if i < 5 else "darkorange"

            ax.set_title(panel_title, fontsize=8.5, color=title_color, fontweight="bold")
            ax.axis("off")

        plt.suptitle(f"Error Analysis Examples — {model_name} (5 FP top, 5 FN bottom)", fontsize=13, fontweight="bold")
        plt.tight_layout()

        contact_fig_path = os.path.join(FIGURES_DIR, f"{slug}_error_examples_OFFICIAL.png")
        plt.savefig(contact_fig_path, dpi=150)
        plt.close()
        log_msg(f"  Saved contact sheet: {contact_fig_path}")

    return error_summary


def build_final_summary(df_model, df_cv):
    """Phần 5: Hợp nhất model_comparison và cross_validation thành final_summary_OFFICIAL.csv."""
    ensure_dir(RESULTS_DIR)

    df_merged = pd.merge(df_model, df_cv, on="model", suffixes=("", "_cv"))

    final_columns = [
        "model",
        "accuracy",
        "precision_fire",
        "recall_fire",
        "f1_fire",
        "specificity",
        "accuracy_mean",
        "accuracy_std",
        "recall_fire_mean",
        "recall_fire_std",
        "f1_fire_mean",
        "f1_fire_std",
        "train_time",
        "prediction_time"
    ]

    df_final = df_merged[final_columns].copy()
    df_final.rename(columns={
        "accuracy_mean": "cv_accuracy_mean",
        "accuracy_std": "cv_accuracy_std",
        "recall_fire_mean": "cv_recall_mean",
        "recall_fire_std": "cv_recall_std",
        "f1_fire_mean": "cv_f1_mean",
        "f1_fire_std": "cv_f1_std"
    }, inplace=True)

    df_final.to_csv(FINAL_SUMMARY_CSV_PATH, index=False, encoding="utf-8")
    log_msg(f"Final summary CSV saved: {FINAL_SUMMARY_CSV_PATH}")

    return df_final


def run_final_evaluation_pipeline():
    """Pipeline tổng hợp đánh giá và phân tích lỗi đầy đủ 6 bước."""
    pipeline_start_time = time.time()

    print("=" * 60, flush=True)
    print("FINAL EVALUATION & ERROR ANALYSIS", flush=True)
    print("=" * 60, flush=True)

    # 1. Loading results
    log_msg("[1/6] Loading official results...")
    t0 = time.time()
    df_model, df_cv = load_official_csv_results()
    log_msg(f"✓ Completed - {time.time() - t0:.2f}s")

    # 2. Creating comparison charts
    log_msg("\n[2/6] Creating model comparison charts...")
    t0 = time.time()
    create_model_comparison_charts(df_model)
    log_msg(f"✓ Completed - {time.time() - t0:.2f}s")
    print("Progress: 33.3%", flush=True)

    # 3. Creating cross-validation charts
    log_msg("\n[3/6] Creating cross-validation charts...")
    t0 = time.time()
    create_cross_validation_charts(df_cv)
    log_msg(f"✓ Completed - {time.time() - t0:.2f}s")
    print("Progress: 50.0%", flush=True)

    # 4. Creating confusion matrices
    log_msg("\n[4/6] Creating confusion matrices...")
    t0 = time.time()
    create_confusion_matrices(df_model)
    log_msg(f"✓ Completed - {time.time() - t0:.2f}s")

    # 5. Error analysis (SVM & Random Forest)
    log_msg("\n[5/6] Performing Error Analysis for SVM & Random Forest...")
    t0 = time.time()
    error_summary = perform_error_analysis()
    log_msg(f"✓ Error Analysis completed - {time.time() - t0:.2f}s")
    print("Progress: 83.3%", flush=True)

    # 6. Building final summary
    log_msg("\n[6/6] Building final summary CSV...")
    t0 = time.time()
    df_final = build_final_summary(df_model, df_cv)
    log_msg(f"✓ Completed - {time.time() - t0:.2f}s")
    print("Progress: 100.0%", flush=True)

    total_runtime = time.time() - pipeline_start_time

    print("\n" + "=" * 60, flush=True)
    print("FINAL EVALUATION COMPLETED", flush=True)
    print(f"Total runtime: {total_runtime:.2f}s", flush=True)
    print("=" * 60 + "\n", flush=True)

    return error_summary, df_final


if __name__ == "__main__":
    run_final_evaluation_pipeline()
