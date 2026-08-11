"""
feature_compare_official.py - Official Feature Set Comparison (HOG vs Color Histogram vs Combined).

MỤC ĐÍCH:
So sánh ảnh hưởng của 3 bộ đặc trưng (HOG 1.764d, Color Histogram 4.096d, Combined 5.860d)
bằng mô hình SVM RBF trên tập dữ liệu CHÍNH THỨC (Hold-out Evaluation ONLY).

DỮ LIỆU:
- TRAIN (15.609 mẫu): data/processed/train/X_{hog,color,combined}.npy, y.npy
- TEST (6.122 mẫu): data/processed/test/X_{hog,color,combined}.npy, y.npy

LƯU Ý BẮT BUỘC:
- CHỈ sử dụng SVM RBF với Pipeline([StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")]).
- Không chạy Cross-Validation, không GridSearch, không thay đổi hyperparameter.
- Fit StandardScaler và SVM CHỈ TRÊN TẬP TRAIN, không fit trên TEST.
"""

import os
import sys
import time
import csv
import numpy as np
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
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from src.utils import (
    RESULTS_DIR, FIGURES_DIR, RANDOM_SEED, ensure_dir, setup_logger
)

logger = setup_logger("feature_compare_official")

# Đường dẫn dữ liệu CHÍNH THỨC
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "train")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "test")

# Đường dẫn xuất tệp kết quả và hình ảnh CHÍNH THỨC
OFFICIAL_FEATURE_RESULTS_PATH = os.path.join(RESULTS_DIR, "feature_comparison_OFFICIAL.csv")
OFFICIAL_F1_FIG_PATH = os.path.join(FIGURES_DIR, "feature_f1_comparison_OFFICIAL.png")
OFFICIAL_RECALL_FIG_PATH = os.path.join(FIGURES_DIR, "feature_recall_comparison_OFFICIAL.png")
OFFICIAL_RUNTIME_FIG_PATH = os.path.join(FIGURES_DIR, "feature_runtime_comparison_OFFICIAL.png")


def log_msg(msg):
    """In log trực tiếp ra terminal với timestamp và flush ngay lập tức."""
    timestamp = time.strftime("[%H:%M:%S]")
    print(f"{timestamp} {msg}", flush=True)


def get_svm_pipeline():
    """Khởi tạo Pipeline StandardScaler + SVM RBF tiêu chuẩn."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight="balanced",
            probability=False,
            cache_size=1000,  # 1GB cache giúp tăng tốc độ train trên 15.6k mẫu
            random_state=RANDOM_SEED
        ))
    ])


def load_dataset_for_feature(feature_file_name):
    """Load X_train, y_train, X_test, y_test cho bộ đặc trưng chỉ định."""
    train_x_path = os.path.join(TRAIN_DIR, feature_file_name)
    train_y_path = os.path.join(TRAIN_DIR, "y.npy")
    test_x_path = os.path.join(TEST_DIR, feature_file_name)
    test_y_path = os.path.join(TEST_DIR, "y.npy")

    if not os.path.exists(train_x_path) or not os.path.exists(test_x_path):
        raise FileNotFoundError(f"Không tìm thấy file đặc trưng {feature_file_name} trong TRAIN hoặc TEST!")

    X_train = np.load(train_x_path)
    y_train = np.load(train_y_path)
    X_test = np.load(test_x_path)
    y_test = np.load(test_y_path)

    return X_train, y_train, X_test, y_test


def run_official_feature_comparison():
    """
    Thực thi so sánh 3 bộ đặc trưng (HOG, Color Histogram, Combined) trên tập CHÍNH THỨC.
    """
    total_start_time = time.time()

    print("=" * 60, flush=True)
    print("OFFICIAL FEATURE COMPARISON", flush=True)
    print("=" * 60, flush=True)

    # Load y_train & y_test để kiểm tra số lượng mẫu
    y_train_check = np.load(os.path.join(TRAIN_DIR, "y.npy"))
    y_test_check = np.load(os.path.join(TEST_DIR, "y.npy"))

    print("\nDataset:", flush=True)
    print(f"TRAIN = {len(y_train_check):,}", flush=True)
    print(f"TEST  = {len(y_test_check):,}\n", flush=True)

    feature_configs = [
        ("HOG", "X_hog.npy", 1764),
        ("Color Histogram", "X_color.npy", 4096),
        ("Combined", "X_combined.npy", 5860)
    ]

    results = []
    completed_times = []

    total_steps = len(feature_configs)

    for idx, (feat_name, feat_file, num_features) in enumerate(feature_configs, 1):
        step_start_time = time.time()

        print("=" * 60, flush=True)
        print(f"[{idx}/{total_steps}] {feat_name.upper()} — {num_features} features", flush=True)
        print("-" * 60, flush=True)

        # 1. Loading data
        log_msg(f"Loading {feat_name}...")
        X_tr, y_tr, X_te, y_te = load_dataset_for_feature(feat_file)
        
        # Verify shape integrity
        assert X_tr.shape[0] == 15609, f"Lỗi số mẫu X_train {feat_name}: {X_tr.shape[0]}"
        assert X_te.shape[0] == 6122, f"Lỗi số mẫu X_test {feat_name}: {X_te.shape[0]}"
        assert X_tr.shape[1] == num_features, f"Lỗi số chiều X_train {feat_name}: {X_tr.shape[1]}"
        assert X_te.shape[1] == num_features, f"Lỗi số chiều X_test {feat_name}: {X_te.shape[1]}"

        # 2. Training
        log_msg(f"Training SVM...")
        pipeline = get_svm_pipeline()
        t0_train = time.time()
        pipeline.fit(X_tr, y_tr)
        train_time = time.time() - t0_train
        log_msg(f"Training completed — {train_time:.1f}s")

        # 3. Predicting
        log_msg(f"Predicting TEST...")
        t0_pred = time.time()
        y_pred = pipeline.predict(X_te)
        predict_time = time.time() - t0_pred
        log_msg(f"Prediction completed — {predict_time:.1f}s")

        # 4. Evaluating
        log_msg(f"Evaluating...")
        cm = confusion_matrix(y_te, y_pred)
        tn, fp, fn, tp = cm.ravel()

        acc = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_te, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_te, y_pred, pos_label=1, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        step_total_time = time.time() - step_start_time
        completed_times.append(step_total_time)

        log_msg(f"{feat_name} completed — Total: {step_total_time:.1f}s")

        print("\nMetrics:", flush=True)
        print(f"Accuracy  = {acc:.4f}", flush=True)
        print(f"Precision = {prec:.4f}", flush=True)
        print(f"Recall    = {rec:.4f}", flush=True)
        print(f"F1        = {f1:.4f}", flush=True)
        print(f"Specificity = {specificity:.4f}", flush=True)
        print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}", flush=True)

        # Progress calculation
        progress_pct = (idx / total_steps) * 100.0
        avg_time_per_feature = np.mean(completed_times)
        remaining_steps = total_steps - idx

        print(f"\nProgress: {progress_pct:.1f}%", flush=True)
        print(f"Average time/feature: {avg_time_per_feature:.1f}s", flush=True)
        if remaining_steps > 0:
            est_remaining_sec = avg_time_per_feature * remaining_steps
            est_remaining_min = est_remaining_sec / 60.0
            print(f"Estimated remaining time: ~{est_remaining_min:.1f} minutes\n", flush=True)
        else:
            print("Estimated remaining time: 0 minutes\n", flush=True)

        results.append({
            "feature_set": feat_name,
            "num_features": num_features,
            "accuracy": round(float(acc), 4),
            "precision_fire": round(float(prec), 4),
            "recall_fire": round(float(rec), 4),
            "f1_fire": round(float(f1), 4),
            "specificity": round(float(specificity), 4),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "train_time": round(float(train_time), 4),
            "prediction_time": round(float(predict_time), 4),
            "total_time": round(float(step_total_time), 4),
            "dataset_status": "REAL_DATASET_OFFICIAL"
        })

    total_runtime = time.time() - total_start_time

    # Xuất kết quả CSV
    ensure_dir(RESULTS_DIR)
    fieldnames = [
        "feature_set", "num_features",
        "accuracy", "precision_fire", "recall_fire", "f1_fire", "specificity",
        "tp", "tn", "fp", "fn",
        "train_time", "prediction_time", "total_time",
        "dataset_status"
    ]

    with open(OFFICIAL_FEATURE_RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    log_msg(f"Saving result to CSV: {OFFICIAL_FEATURE_RESULTS_PATH}")

    # Tạo biểu đồ so sánh
    plot_official_feature_comparisons(results)

    print("=" * 60, flush=True)
    print("FEATURE COMPARISON COMPLETED", flush=True)
    print(f"Total runtime: {total_runtime:.1f}s ({total_runtime/60.0:.2f} mins)", flush=True)
    print("=" * 60, flush=True)

    # In bảng tổng hợp
    print_summary_table(results)

    return results


def plot_official_feature_comparisons(results):
    """
    Tạo 3 biểu đồ so sánh F1-score, Recall và Runtime từ kết quả đã tính.
    """
    ensure_dir(FIGURES_DIR)

    feat_names = [r["feature_set"] for r in results]
    x = np.arange(len(feat_names))
    width = 0.45

    # 1. Biểu đồ F1-score comparison
    f1_scores = [r["f1_fire"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x, f1_scores, width, color=["#2196F3", "#FF9800", "#4CAF50"], edgecolor="black", linewidth=1)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.01, f"{h:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("F1 Score (Fire Class)", fontsize=11, fontweight="bold")
    ax.set_title("Official Feature Comparison - F1 Score (Fire Class)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(feat_names, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OFFICIAL_F1_FIG_PATH, dpi=150)
    plt.close()
    log_msg(f"Saved F1 chart: {OFFICIAL_F1_FIG_PATH}")

    # 2. Biểu đồ Recall comparison
    recall_scores = [r["recall_fire"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars2 = ax.bar(x, recall_scores, width, color=["#9C27B0", "#E91E63", "#00BCD4"], edgecolor="black", linewidth=1)

    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.01, f"{h:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Recall Score (Fire Class)", fontsize=11, fontweight="bold")
    ax.set_title("Official Feature Comparison - Recall Score (Fire Class)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(feat_names, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OFFICIAL_RECALL_FIG_PATH, dpi=150)
    plt.close()
    log_msg(f"Saved Recall chart: {OFFICIAL_RECALL_FIG_PATH}")

    # 3. Biểu đồ Runtime comparison (Train, Predict, Total)
    train_times = [r["train_time"] for r in results]
    predict_times = [r["prediction_time"] for r in results]
    total_times = [r["total_time"] for r in results]

    bar_w = 0.25
    x_rt = np.arange(len(feat_names))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    b_tr = ax.bar(x_rt - bar_w, train_times, bar_w, label="Train Time (s)", color="#42A5F5")
    b_pr = ax.bar(x_rt, predict_times, bar_w, label="Predict Time (s)", color="#FFA726")
    b_tot = ax.bar(x_rt + bar_w, total_times, bar_w, label="Total Time (s)", color="#66BB6A")

    for bars in [b_tr, b_pr, b_tot]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + (max(total_times) * 0.01), f"{h:.1f}s", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax.set_title("Official Feature Comparison - Execution Time (Seconds)", fontsize=13, fontweight="bold")
    ax.set_xticks(x_rt)
    ax.set_xticklabels(feat_names, fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(total_times) * 1.18)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OFFICIAL_RUNTIME_FIG_PATH, dpi=150)
    plt.close()
    log_msg(f"Saved Runtime chart: {OFFICIAL_RUNTIME_FIG_PATH}")


def print_summary_table(results):
    """In bảng tổng hợp kết quả ra terminal."""
    print("\n" + "=" * 115, flush=True)
    print("BẢNG TỔNG HỢP KẾT QUẢ SO SÁNH BỘ ĐẶC TRƯNG CHÍNH THỨC (HOLD-OUT EVALUATION)", flush=True)
    print("=" * 115, flush=True)
    print(f"{'Bộ Đặc Trưng':<18} | {'Số Chiều':<8} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Specificity':<11} | {'Train(s)':<9} | {'Pred(s)':<8} | {'Total(s)':<9}", flush=True)
    print("-" * 115, flush=True)
    for r in results:
        print(f"{r['feature_set']:<18} | {r['num_features']:<8} | {r['accuracy']:<10.4f} | {r['precision_fire']:<10.4f} | {r['recall_fire']:<10.4f} | {r['f1_fire']:<10.4f} | {r['specificity']:<11.4f} | {r['train_time']:<9.1f} | {r['prediction_time']:<8.1f} | {r['total_time']:<9.1f}", flush=True)
    print("=" * 115 + "\n", flush=True)


if __name__ == "__main__":
    run_official_feature_comparison()
