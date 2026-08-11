"""
advanced_analysis_official.py - Advanced Academic Analysis cho BTL Data Mining.

MỤC ĐÍCH:
1. ROC Curve & AUC: Đánh giá khả năng phân biệt Fire / Non-fire của 4 mô hình trên tập TEST.
2. PCA Visualization: Trực quan hóa 2D không gian đặc trưng HOG + Color Histogram (5.860 chiều) trên tập TRAIN.
3. McNemar Test: Kiểm tra ý nghĩa thống kê về sự khác biệt hiệu năng giữa SVM và Random Forest trên tập TEST.
4. Tổng hợp các báo cáo nâng cao (CSV và JSON).

LƯU Ý: KHÔNG TRAIN LẠI BẤT KỲ MODEL NÀO. Tái sử dụng 100% tài nguyên đã huấn luyện.
"""

import os
import sys
import time
import csv
import json
import pickle
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")  # Backend hiển thị không cần GUI
import matplotlib.pyplot as plt

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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score

from src.utils import (
    RESULTS_DIR, FIGURES_DIR, MODELS_DIR, RANDOM_SEED, ensure_dir, setup_logger
)

logger = setup_logger("advanced_analysis_official")

# Đường dẫn tệp đầu vào
TRAIN_X_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train", "X_combined.npy")
TRAIN_Y_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train", "y.npy")
TEST_X_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "test", "X_combined.npy")
TEST_Y_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "test", "y.npy")

OFFICIAL_MODELS_DIR = os.path.join(MODELS_DIR, "official")
MODEL_PATHS = {
    "Logistic Regression": os.path.join(OFFICIAL_MODELS_DIR, "logistic_regression_combined.pkl"),
    "K-Nearest Neighbors": os.path.join(OFFICIAL_MODELS_DIR, "knn_combined.pkl"),
    "SVM": os.path.join(OFFICIAL_MODELS_DIR, "svm_combined.pkl"),
    "Random Forest": os.path.join(OFFICIAL_MODELS_DIR, "random_forest_combined.pkl")
}

# Đường dẫn tệp đầu ra
ROC_CSV_PATH = os.path.join(RESULTS_DIR, "roc_auc_OFFICIAL.csv")
ROC_FIG_PATH = os.path.join(FIGURES_DIR, "roc_curve_OFFICIAL.png")
PCA_CSV_PATH = os.path.join(RESULTS_DIR, "pca_summary_OFFICIAL.csv")
PCA_FIG_PATH = os.path.join(FIGURES_DIR, "pca_feature_space_OFFICIAL.png")
MCNEMAR_CSV_PATH = os.path.join(RESULTS_DIR, "mcnemar_svm_vs_rf_OFFICIAL.csv")
ADVANCED_CSV_PATH = os.path.join(RESULTS_DIR, "advanced_analysis_OFFICIAL.csv")
ADVANCED_JSON_PATH = os.path.join(RESULTS_DIR, "advanced_analysis_summary_OFFICIAL.json")


def log_msg(msg):
    """In log trực tiếp ra terminal với timestamp và flush ngay lập tức."""
    timestamp = time.strftime("[%H:%M:%S]")
    print(f"{timestamp} {msg}", flush=True)


def run_roc_auc_analysis(X_test, y_test):
    """PHẦN A: ROC Curve & AUC Analysis cho cả 4 mô hình trên tập TEST."""
    log_msg("--- PHẦN A: ROC CURVE & AUC ANALYSIS ---")

    roc_results = []
    plot_data = {}

    model_colors = {
        "Logistic Regression": "#42A5F5",
        "K-Nearest Neighbors": "#AB47BC",
        "SVM": "#FFA726",
        "Random Forest": "#66BB6A"
    }

    t0_all = time.time()

    for model_name, model_path in MODEL_PATHS.items():
        log_msg(f"  Loading {model_name} from {os.path.basename(model_path)}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy model {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        log_msg(f"  Computing decision scores for {model_name}...")
        t0_score = time.time()

        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
            score_method = "predict_proba"
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            score_method = "decision_function"
        else:
            raise AttributeError(f"Model {model_name} không hỗ trợ predict_proba hay decision_function!")

        score_time = time.time() - t0_score
        auc_val = roc_auc_score(y_test, scores)
        fpr, tpr, _ = roc_curve(y_test, scores)

        log_msg(f"  ✓ {model_name} completed in {score_time:.1f}s — AUC: {auc_val:.4f} (method: {score_method})")

        roc_results.append({
            "model": model_name,
            "roc_auc": round(float(auc_val), 4),
            "score_method": score_method,
            "dataset_status": "REAL_DATASET_OFFICIAL"
        })

        plot_data[model_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc_val,
            "color": model_colors[model_name]
        }

    # 1. Save ROC CSV
    ensure_dir(RESULTS_DIR)
    fieldnames = ["model", "roc_auc", "score_method", "dataset_status"]
    with open(ROC_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(roc_results)
    log_msg(f"  Saved ROC-AUC CSV: {ROC_CSV_PATH}")

    # 2. Draw ROC Curves Plot
    ensure_dir(FIGURES_DIR)
    fig, ax = plt.subplots(figsize=(8, 6.5))

    for model_name, data in plot_data.items():
        ax.plot(data["fpr"], data["tpr"], color=data["color"], lw=2.2,
                label=f"{model_name} (AUC = {data['auc']:.4f})")

    # Baseline random classifier line
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11, fontweight="bold")
    ax.set_title("Official ROC Curves Comparison (TEST Dataset)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(ROC_FIG_PATH, dpi=150)
    plt.close()
    log_msg(f"  Saved ROC Curve plot: {ROC_FIG_PATH}")

    total_roc_time = time.time() - t0_all
    return roc_results, total_roc_time


def run_pca_visualization(X_train, y_train):
    """PHẦN B: PCA 2D Feature Space Visualization trên tập TRAIN."""
    log_msg("\n--- PHẦN B: PCA VISUALIZATION (2D) ---")
    t0_pca = time.time()

    n_samples, n_dims = X_train.shape
    log_msg(f"  Input TRAIN feature matrix: {n_samples:,} samples × {n_dims:,} dimensions")

    log_msg("  Standardizing TRAIN features with StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    log_msg("  Fitting PCA(n_components=2, random_state=42)...")
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)

    ev_ratio = pca.explained_variance_ratio_
    pc1_var = float(ev_ratio[0])
    pc2_var = float(ev_ratio[1])
    total_var = float(np.sum(ev_ratio))

    log_msg(f"  ✓ PCA completed in {time.time() - t0_pca:.2f}s")
    log_msg(f"  PC1 Explained Variance: {pc1_var*100:.2f}%")
    log_msg(f"  PC2 Explained Variance: {pc2_var*100:.2f}%")
    log_msg(f"  Total Explained Variance (2D): {total_var*100:.2f}%")

    # 1. Save PCA Summary CSV
    ensure_dir(RESULTS_DIR)
    pca_results = {
        "samples_used": n_samples,
        "original_dimensions": n_dims,
        "pc1_explained_variance": round(pc1_var, 6),
        "pc2_explained_variance": round(pc2_var, 6),
        "total_explained_variance": round(total_var, 6)
    }

    with open(PCA_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pca_results.keys()))
        writer.writeheader()
        writer.writerow(pca_results)
    log_msg(f"  Saved PCA Summary CSV: {PCA_CSV_PATH}")

    # 2. Plot 2D PCA Feature Space
    ensure_dir(FIGURES_DIR)
    fig, ax = plt.subplots(figsize=(9, 6.5))

    non_fire_mask = (y_train == 0)
    fire_mask = (y_train == 1)

    ax.scatter(X_pca[non_fire_mask, 0], X_pca[non_fire_mask, 1],
               c="#1E88E5", alpha=0.35, s=12, label="Non-fire (Class 0)", edgecolors="none")
    ax.scatter(X_pca[fire_mask, 0], X_pca[fire_mask, 1],
               c="#E53935", alpha=0.35, s=12, label="Fire (Class 1)", edgecolors="none")

    ax.set_xlabel(f"Principal Component 1 ({pc1_var*100:.2f}% Variance)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Principal Component 2 ({pc2_var*100:.2f}% Variance)", fontsize=11, fontweight="bold")
    ax.set_title("PCA 2D Feature Space Visualization (HOG + Color Histogram)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Subtitle annotation box
    ax.text(0.02, 0.95, f"Total 2D Explained Variance: {total_var*100:.2f}%\nN = {n_samples:,} TRAIN samples",
            transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="gray"))

    plt.tight_layout()
    plt.savefig(PCA_FIG_PATH, dpi=150)
    plt.close()
    log_msg(f"  Saved PCA Feature Space plot: {PCA_FIG_PATH}")

    total_pca_time = time.time() - t0_pca
    return pca_results, total_pca_time


def run_mcnemar_test(X_test, y_test):
    """PHẦN C: McNemar Test kiểm tra sự khác biệt giữa SVM và Random Forest trên tập TEST."""
    log_msg("\n--- PHẦN C: MCNEMAR TEST (SVM vs RANDOM FOREST) ---")
    t0_mc = time.time()

    svm_path = MODEL_PATHS["SVM"]
    rf_path = MODEL_PATHS["Random Forest"]

    log_msg("  Loading saved SVM and Random Forest models...")
    with open(svm_path, "rb") as f:
        svm = pickle.load(f)
    with open(rf_path, "rb") as f:
        rf = pickle.load(f)

    log_msg("  Predicting TEST set with SVM & Random Forest...")
    y_pred_svm = svm.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    svm_correct = (y_pred_svm == y_test)
    rf_correct = (y_pred_rf == y_test)

    both_correct = int(np.sum(svm_correct & rf_correct))
    svm_correct_rf_wrong = int(np.sum(svm_correct & ~rf_correct))  # b
    svm_wrong_rf_correct = int(np.sum(~svm_correct & rf_correct))  # c
    both_wrong = int(np.sum(~svm_correct & ~rf_correct))

    b = svm_correct_rf_wrong
    c = svm_wrong_rf_correct

    log_msg(f"  Contingency Table:")
    log_msg(f"    - Both Correct  (a): {both_correct:,}")
    log_msg(f"    - SVM Correct, RF Wrong (b): {b}")
    log_msg(f"    - SVM Wrong, RF Correct (c): {c}")
    log_msg(f"    - Both Wrong    (d): {both_wrong}")

    # Tính toán McNemar Chi-Square Statistic với hiệu chỉnh liên tục (Continuity Correction)
    if (b + c) > 0:
        stat = float(((abs(b - c) - 1.0) ** 2) / (b + c))
        p_val = float(stats.chi2.sf(stat, df=1))
        p_exact = float(stats.binomtest(b, b + c, 0.5).pvalue)
    else:
        stat = 0.0
        p_val = 1.0
        p_exact = 1.0

    alpha = 0.05
    is_significant = bool(p_val < alpha)

    if is_significant:
        conclusion = f"Khac biet co y nghia thong ke giua SVM va Random Forest tren tap TEST (p = {p_val:.4f} < {alpha})."
    else:
        conclusion = f"Chua du bang chung thong ke de khang dinh su khac biet giua SVM va Random Forest tren tap TEST (p = {p_val:.4f} >= {alpha})."

    log_msg(f"  Chi-Square Statistic: {stat:.4f}")
    log_msg(f"  p-value (Chi2 continuity corrected): {p_val:.4f}")
    log_msg(f"  p-value (Exact Binomial): {p_exact:.4f}")
    log_msg(f"  Significance level (alpha): {alpha}")
    log_msg(f"  Is Significant: {is_significant}")
    log_msg(f"  Conclusion: {conclusion}")

    # 1. Save McNemar CSV
    ensure_dir(RESULTS_DIR)
    mcnemar_results = {
        "both_correct": both_correct,
        "svm_correct_rf_wrong": b,
        "svm_wrong_rf_correct": c,
        "both_wrong": both_wrong,
        "statistic": round(stat, 4),
        "p_value": round(p_val, 4),
        "alpha": alpha,
        "significant": is_significant,
        "conclusion": conclusion
    }

    with open(MCNEMAR_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(mcnemar_results.keys()))
        writer.writeheader()
        writer.writerow(mcnemar_results)
    log_msg(f"  Saved McNemar Test CSV: {MCNEMAR_CSV_PATH}")

    total_mc_time = time.time() - t0_mc
    return mcnemar_results, total_mc_time


def build_advanced_summary_files(roc_results, pca_results, mcnemar_results):
    """PHẦN D: Tạo advanced_analysis_OFFICIAL.csv và advanced_analysis_summary_OFFICIAL.json."""
    ensure_dir(RESULTS_DIR)

    # 1. Advanced Analysis CSV
    with open(ADVANCED_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "roc_auc", "score_method", "dataset_status"])
        writer.writeheader()
        writer.writerows(roc_results)
    log_msg(f"Saved Advanced Analysis CSV: {ADVANCED_CSV_PATH}")

    # 2. Advanced Analysis JSON
    summary_json = {
        "roc_auc_scores": roc_results,
        "pca_summary": pca_results,
        "mcnemar_test": mcnemar_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(ADVANCED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)
    log_msg(f"Saved Advanced Analysis JSON: {ADVANCED_JSON_PATH}")


def run_advanced_analysis_pipeline():
    """Pipeline tổng hợp 3 phân tích học thuật nâng cao."""
    pipeline_start_time = time.time()

    print("=" * 60, flush=True)
    print("ADVANCED ACADEMIC ANALYSIS", flush=True)
    print("=" * 60, flush=True)

    log_msg("Loading TEST and TRAIN feature matrices...")
    X_test = np.load(TEST_X_PATH)
    y_test = np.load(TEST_Y_PATH)
    X_train = np.load(TRAIN_X_PATH)
    y_train = np.load(TRAIN_Y_PATH)

    log_msg(f"Loaded TRAIN matrix: {X_train.shape}, TEST matrix: {X_test.shape}")

    # [1/3] ROC-AUC Analysis
    log_msg("\n[1/3] ROC-AUC Analysis...")
    t0 = time.time()
    roc_results, roc_time = run_roc_auc_analysis(X_test, y_test)
    elapsed_1 = time.time() - pipeline_start_time
    print(f"Progress: 33.3% | Elapsed: {elapsed_1:.1f}s | ETA: ~{elapsed_1 * 2:.1f}s", flush=True)

    # [2/3] PCA Visualization
    log_msg("\n[2/3] PCA Visualization...")
    t0 = time.time()
    pca_results, pca_time = run_pca_visualization(X_train, y_train)
    elapsed_2 = time.time() - pipeline_start_time
    print(f"Progress: 66.7% | Elapsed: {elapsed_2:.1f}s | ETA: ~{(elapsed_2 / 2):.1f}s", flush=True)

    # [3/3] McNemar Test
    log_msg("\n[3/3] McNemar Test (SVM vs Random Forest)...")
    t0 = time.time()
    mcnemar_results, mc_time = run_mcnemar_test(X_test, y_test)
    elapsed_3 = time.time() - pipeline_start_time
    print(f"Progress: 100.0% | Elapsed: {elapsed_3:.1f}s", flush=True)

    # Summary outputs
    log_msg("\nBuilding advanced summary files...")
    build_advanced_summary_files(roc_results, pca_results, mcnemar_results)

    total_runtime = time.time() - pipeline_start_time

    print("\n" + "=" * 60, flush=True)
    print("ADVANCED ANALYSIS COMPLETED", flush=True)
    print(f"Total runtime: {total_runtime:.2f}s ({total_runtime/60.0:.2f} mins)", flush=True)
    print("=" * 60 + "\n", flush=True)

    return roc_results, pca_results, mcnemar_results, total_runtime


if __name__ == "__main__":
    run_advanced_analysis_pipeline()
