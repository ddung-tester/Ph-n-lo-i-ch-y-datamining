"""
train_cv_official.py - Official 5-Fold Stratified Cross-Validation trên TRAIN DATASET ONLY.

MỤC ĐÍCH:
Đánh giá độ ổn định và khả năng tổng quát hóa của 4 mô hình Machine Learning (Logistic Regression, KNN, SVM RBF, Random Forest)
bằng 5-Fold Stratified Cross-Validation CHỈ TRÊN TẬP TRAIN DATASET (15.609 mẫu, 5.860 chiều).

LƯU Ý BẮT BUỘC:
- Tuyệt đối không nạp hay sử dụng tập TEST dataset trong script này.
- StandardScaler bắt buộc nằm trong sklearn Pipeline để fit riêng trên từng training fold.
"""

import os
import sys
import time
import csv
import numpy as np

# Reconfigure stdout cho Windows console để hiển thị UTF-8 không bị lỗi
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Đảm bảo PROJECT_ROOT trong sys.path để hỗ trợ chạy trực tiếp
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

from src.feature_cache import load_cache, check_cache_exists, build_feature_cache
from src.utils import (
    RESULTS_DIR, RANDOM_SEED, ensure_dir, setup_logger
)

logger = setup_logger("train_cv_official")

# Đường dẫn xuất file kết quả Cross-Validation chính thức và Hold-out chính thức
CV_OFFICIAL_RESULTS_PATH = os.path.join(RESULTS_DIR, "cross_validation_OFFICIAL.csv")
HOLDOUT_OFFICIAL_RESULTS_PATH = os.path.join(RESULTS_DIR, "model_comparison_OFFICIAL.csv")


def get_official_models():
    """
    Khởi tạo 4 mô hình Machine Learning Baseline chính thức theo đúng cấu hình yêu cầu.
    """
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_SEED
            ))
        ]),
        
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=5
            ))
        ]),
        
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                class_weight="balanced",
                probability=False,
                cache_size=1000,  # 1GB cache gia tăng tốc độ tính toán cho 5 folds
                random_state=RANDOM_SEED
            ))
        ]),
        
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    }
    
    return models


def run_official_cross_validation():
    """
    Thực thi 5-Fold Stratified Cross-Validation duy nhất trên tập TRAIN chính thức.
    """
    print("=" * 95)
    print("BƯỚC 6: OFFICIAL 5-FOLD STRATIFIED CROSS-VALIDATION (CHỈ TRÊN TRAIN DATASET)")
    print("=" * 95)
    
    # 1. Nạp cache dữ liệu TRAIN CHÍNH THỨC
    if not check_cache_exists("train"):
        logger.info("Chưa có feature cache tập TRAIN. Tiến hành xây dựng cache...")
        build_feature_cache()
        
    _, _, X_train, y_train = load_cache("train")
    
    # 2. Validation dữ liệu TRAIN
    assert X_train.shape == (15609, 5860), f"Lỗi X_train shape: {X_train.shape} != (15609, 5860)"
    assert y_train.shape == (15609,), f"Lỗi y_train shape: {y_train.shape} != (15609,)"
    assert not np.isnan(X_train).any(), "Phát hiện NaN trong X_train!"
    assert not np.isinf(X_train).any(), "Phát hiện Inf trong X_train!"
    assert X_train.dtype == np.float32, f"X_train dtype {X_train.dtype} != float32"
    
    logger.info("✓ XÁC NHẬN: Cross-Validation CHỈ sử dụng tập TRAIN:")
    logger.info(f"  - Tổng mẫu TRAIN: {X_train.shape[0]} mẫu, {X_train.shape[1]} chiều đặc trưng")
    logger.info(f"  - Phân bố nhãn: fire (label 1) = {np.sum(y_train==1)}, non_fire (label 0) = {np.sum(y_train==0)}")
    logger.info("  - Tập TEST được giữ độc lập hoàn toàn, 0% rò rỉ dữ liệu.\n")
    
    ensure_dir(RESULTS_DIR)
    
    # 3. Khởi tạo StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    models = get_official_models()
    results = []
    
    # 4. Huấn luyện 5-Fold Cross-Validation cho từng mô hình
    for model_name, model in models.items():
        logger.info(f"===========================================================")
        logger.info(f"BẮT ĐẦU 5-FOLD CV CHO MÔ HÌNH: {model_name.upper()}")
        logger.info(f"===========================================================")
        
        fold_accs = []
        fold_precs = []
        fold_recs = []
        fold_f1s = []
        
        t_model_start = time.time()
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            
            t_fold_start = time.time()
            
            # Clone/Fit pipeline trên fold train_idx
            # StandardScaler tự động fit CHỈ TRÊN X_tr
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            
            fold_time = time.time() - t_fold_start
            
            # Tính metrics trên val_idx
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
            rec = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
            f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
            
            fold_accs.append(acc)
            fold_precs.append(prec)
            fold_recs.append(rec)
            fold_f1s.append(f1)
            
            logger.info(f"  [Fold {fold}/5] Time: {fold_time:.2f}s | Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
            
        cv_total_time = time.time() - t_model_start
        
        # Tính Mean và Std
        acc_mean, acc_std = np.mean(fold_accs), np.std(fold_accs)
        prec_mean, prec_std = np.mean(fold_precs), np.std(fold_precs)
        rec_mean, rec_std = np.mean(fold_recs), np.std(fold_recs)
        f1_mean, f1_std = np.mean(fold_f1s), np.std(fold_f1s)
        
        res_dict = {
            "model": model_name,
            "accuracy_mean": round(float(acc_mean), 4),
            "accuracy_std": round(float(acc_std), 4),
            "precision_fire_mean": round(float(prec_mean), 4),
            "precision_fire_std": round(float(prec_std), 4),
            "recall_fire_mean": round(float(rec_mean), 4),
            "recall_fire_std": round(float(rec_std), 4),
            "f1_fire_mean": round(float(f1_mean), 4),
            "f1_fire_std": round(float(f1_std), 4),
            "cv_time": round(float(cv_total_time), 4),
            "dataset_status": "REAL_DATASET_OFFICIAL"
        }
        results.append(res_dict)
        
        logger.info(f"\n✓ TỔNG HỢP CV 5-FOLD [{model_name}]:")
        logger.info(f"  - Accuracy:       {acc_mean:.4f} ± {acc_std:.4f}")
        logger.info(f"  - Precision fire: {prec_mean:.4f} ± {prec_std:.4f}")
        logger.info(f"  - Recall fire:    {rec_mean:.4f} ± {rec_std:.4f}")
        logger.info(f"  - F1-Score fire:  {f1_mean:.4f} ± {f1_std:.4f}")
        logger.info(f"  - Tổng thời gian CV: {cv_total_time:.2f} giây\n")
        
    # 5. Lưu kết quả ra file CSV
    fieldnames = [
        "model", "accuracy_mean", "accuracy_std",
        "precision_fire_mean", "precision_fire_std",
        "recall_fire_mean", "recall_fire_std",
        "f1_fire_mean", "f1_fire_std",
        "cv_time", "dataset_status"
    ]
    
    with open(CV_OFFICIAL_RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    logger.info(f"Đã xuất file kết quả Cross-Validation chính thức: {CV_OFFICIAL_RESULTS_PATH}")
    
    # 6. In bảng tổng hợp CV và so sánh trực tiếp với Hold-out
    print_summary_and_holdout_comparison(results)
    
    return results


def load_holdout_results():
    """Nạp kết quả Hold-out từ model_comparison_OFFICIAL.csv nếu tồn tại."""
    if not os.path.isfile(HOLDOUT_OFFICIAL_RESULTS_PATH):
        return {}
        
    holdout_map = {}
    with open(HOLDOUT_OFFICIAL_RESULTS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            holdout_map[row["model"]] = {
                "accuracy": float(row["accuracy"]),
                "precision_fire": float(row["precision_fire"]),
                "recall_fire": float(row["recall_fire"]),
                "f1_fire": float(row["f1_fire"])
            }
    return holdout_map


def print_summary_and_holdout_comparison(cv_results):
    """In bảng kết quả CV 5-Fold và so sánh với Hold-out."""
    print("\n" + "=" * 115)
    print("BẢNG TỔNG HỢP KẾT QUẢ 5-FOLD STRATIFIED CROSS-VALIDATION (CHỈ TRÊN TRAIN DATASET)")
    print("=" * 115)
    print(f"{'Mô hình (Model)':<22} | {'Accuracy (Mean±Std)':<22} | {'Precision (Mean±Std)':<22} | {'Recall (Mean±Std)':<22} | {'F1-Score (Mean±Std)':<22} | {'CV Time (s)':<10}")
    print("-" * 125)
    for r in cv_results:
        acc_str = f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}"
        prec_str = f"{r['precision_fire_mean']:.4f} ± {r['precision_fire_std']:.4f}"
        rec_str = f"{r['recall_fire_mean']:.4f} ± {r['recall_fire_std']:.4f}"
        f1_str = f"{r['f1_fire_mean']:.4f} ± {r['f1_fire_std']:.4f}"
        print(f"{r['model']:<22} | {acc_str:<22} | {prec_str:<22} | {rec_str:<22} | {f1_str:<22} | {r['cv_time']:<10.2f}")
    print("=" * 115)
    
    # Nạp dữ liệu Hold-out để so sánh
    holdout_map = load_holdout_results()
    
    if holdout_map:
        print("\n" + "=" * 120)
        print("BẢNG SO SÁNH TRỰC TIẾP GIỮA HOLD-OUT EVALUATION (TEST SET) VÀ 5-FOLD CROSS-VALIDATION (TRAIN SET)")
        print("=" * 120)
        print(f"{'Mô hình (Model)':<22} | {'Hold-out F1':<12} | {'CV F1 Mean':<12} | {'CV F1 Std':<10} | {'Hold-out Rec':<13} | {'CV Rec Mean':<13} | {'CV Rec Std':<10} | {'Chênh lệch F1':<13}")
        print("-" * 125)
        for r in cv_results:
            m = r["model"]
            if m in holdout_map:
                h_f1 = holdout_map[m]["f1_fire"]
                h_rec = holdout_map[m]["recall_fire"]
                cv_f1 = r["f1_fire_mean"]
                cv_f1_std = r["f1_fire_std"]
                cv_rec = r["recall_fire_mean"]
                cv_rec_std = r["recall_fire_std"]
                diff_f1 = abs(h_f1 - cv_f1)
                
                print(f"{m:<22} | {h_f1:<12.4f} | {cv_f1:<12.4f} | {cv_f1_std:<10.4f} | {h_rec:<13.4f} | {cv_rec:<13.4f} | {cv_rec_std:<10.4f} | {diff_f1:<13.4f}")
        print("=" * 120)
        
    best_rec_cv = max(cv_results, key=lambda x: x["recall_fire_mean"])
    best_f1_cv = max(cv_results, key=lambda x: x["f1_fire_mean"])
    most_stable = min(cv_results, key=lambda x: x["f1_fire_std"])
    
    print("\nĐÁNH GIÁ ỔN ĐỊNH VÀ PHÂN TÍCH CROSS-VALIDATION:")
    print(f"  🏆 Mô hình có CV Recall-fire cao nhất: {best_rec_cv['model']} ({best_rec_cv['recall_fire_mean']:.4f} ± {best_rec_cv['recall_fire_std']:.4f})")
    print(f"  ⭐ Mô hình có CV F1-fire cao nhất:     {best_f1_cv['model']} ({best_f1_cv['f1_fire_mean']:.4f} ± {best_f1_cv['f1_fire_std']:.4f})")
    print(f"  🛡️ Mô hình ổn định nhất (F1 std thấp nhất): {most_stable['model']} (Std = {most_stable['f1_fire_std']:.4f})")
    print("=" * 120 + "\n")


if __name__ == "__main__":
    run_official_cross_validation()
