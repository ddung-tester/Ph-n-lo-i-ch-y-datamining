"""
train_compare.py - Official Baseline Model Comparison trên DATASET THẬT CHÍNH THỨC.

MỤC ĐÍCH:
Huấn luyện và so sánh chính thức 4 thuật toán Machine Learning (Logistic Regression, KNN, SVM RBF, Random Forest)
trên tập dữ liệu thật (TRAIN: 15.609 mẫu, TEST: 6.122 mẫu) với bộ đặc trưng Combined (5.860 chiều).

LƯU Ý BẮT BUỘC:
- Sử dụng trực tiếp TRAIN CACHE cho model.fit() và TEST CACHE cho model.predict().
- Tuyệt đối không dùng train_test_split lại trên toàn bộ dữ liệu.
- Tập TEST được giữ độc lập 100%, không fit bất kỳ StandardScaler hay mô hình nào trên TEST.
"""

import os
import sys
import time
import csv
import pickle
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

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from src.feature_cache import load_cache, check_cache_exists, build_feature_cache
from src.utils import (
    RESULTS_DIR, MODELS_DIR, RANDOM_SEED, ensure_dir, setup_logger
)

logger = setup_logger("train_compare_official")

# Đường dẫn xuất kết quả và lưu model
OFFICIAL_RESULTS_PATH = os.path.join(RESULTS_DIR, "model_comparison_OFFICIAL.csv")
OFFICIAL_MODELS_DIR = os.path.join(MODELS_DIR, "official")


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
                probability=False,  # probability=False để tối ưu thời gian huấn luyện
                cache_size=1000,    # 1GB cache để gia tăng tốc độ fit SVM trên dữ liệu lớn
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


def validate_official_data(X_train, y_train, X_test, y_test):
    """
    Kiểm tra tính toàn vẹn dữ liệu TRAIN và TEST chính thức trước khi huấn luyện.
    """
    logger.info("Đang kiểm tra dữ liệu TRAIN & TEST chính thức...")
    
    # 1. Kiểm tra shape
    assert X_train.shape == (15609, 5860), f"Lỗi X_train shape: {X_train.shape} != (15609, 5860)"
    assert y_train.shape == (15609,), f"Lỗi y_train shape: {y_train.shape} != (15609,)"
    assert X_test.shape == (6122, 5860), f"Lỗi X_test shape: {X_test.shape} != (6122, 5860)"
    assert y_test.shape == (6122,), f"Lỗi y_test shape: {y_test.shape} != (6122,)"
    
    # 2. Kiểm tra NaN và Inf
    assert not np.isnan(X_train).any(), "Phát hiện NaN trong X_train!"
    assert not np.isnan(y_train).any(), "Phát hiện NaN trong y_train!"
    assert not np.isnan(X_test).any(), "Phát hiện NaN trong X_test!"
    assert not np.isnan(y_test).any(), "Phát hiện NaN trong y_test!"
    
    assert not np.isinf(X_train).any(), "Phát hiện Inf trong X_train!"
    assert not np.isinf(X_test).any(), "Phát hiện Inf trong X_test!"
    
    # 3. Kiểm tra dtype
    assert X_train.dtype == np.float32, f"X_train dtype {X_train.dtype} != float32"
    assert X_test.dtype == np.float32, f"X_test dtype {X_test.dtype} != float32"
    
    logger.info("✓ Data Validation CHÍNH THỨC đạt 100%:")
    logger.info(f"  - TRAIN: {X_train.shape[0]} mẫu, {X_train.shape[1]} đặc trưng (fire={np.sum(y_train==1)}, non_fire={np.sum(y_train==0)})")
    logger.info(f"  - TEST:  {X_test.shape[0]} mẫu, {X_test.shape[1]} đặc trưng (fire={np.sum(y_test==1)}, non_fire={np.sum(y_test==0)})")


def run_official_baseline_comparison():
    """
    Thực thi huấn luyện 4 mô hình Baseline trên tập TRAIN chính thức và đánh giá trên tập TEST.
    """
    print("=" * 85)
    print("BƯỚC 5: OFFICIAL BASELINE MODEL COMPARISON (DATASET THẬT CHÍNH THỨC)")
    print("=" * 85)
    
    # 1. Nạp cache dữ liệu TRAIN và TEST tách biệt
    if not check_cache_exists("train") or not check_cache_exists("test"):
        logger.info("Chưa có đầy đủ feature cache. Tiến hành xây dựng cache...")
        build_feature_cache()
        
    _, _, X_train, y_train = load_cache("train")
    _, _, X_test, y_test = load_cache("test")
    
    # 2. Validation dữ liệu
    validate_official_data(X_train, y_train, X_test, y_test)
    
    ensure_dir(OFFICIAL_MODELS_DIR)
    ensure_dir(RESULTS_DIR)
    
    models = get_official_models()
    results = []
    
    # Map tên file model để lưu
    model_filename_map = {
        "Logistic Regression": "logistic_regression_combined.pkl",
        "K-Nearest Neighbors": "knn_combined.pkl",
        "SVM": "svm_combined.pkl",
        "Random Forest": "random_forest_combined.pkl"
    }
    
    # 3. Huấn luyện và đánh giá độc lập từng mô hình
    for model_name, model in models.items():
        logger.info(f"\n---> [OFFICIAL TRAIN] Đang huấn luyện: {model_name}...")
        
        # Đo thời gian huấn luyện trên TRAIN CACHE
        t0 = time.time()
        try:
            model.fit(X_train, y_train)
            train_time = time.time() - t0
        except Exception as e:
            logger.error(f"Lỗi khi fit mô hình {model_name}: {e}")
            continue
            
        # Đo thời gian dự đoán trên TEST CACHE
        logger.info(f"---> [OFFICIAL PREDICT] Đang dự đoán tập TEST: {model_name}...")
        t1 = time.time()
        try:
            y_pred = model.predict(X_test)
            predict_time = time.time() - t1
        except Exception as e:
            logger.error(f"Lỗi khi predict mô hình {model_name}: {e}")
            continue
            
        # Tính toán Confusion Matrix: tn, fp, fn, tp (pos_label = fire = 1)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        # Tính các chỉ số chính thức
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Lưu file model đã train vào models/official/
        save_path = os.path.join(OFFICIAL_MODELS_DIR, model_filename_map[model_name])
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"  └ Đã lưu model: {save_path}")
        
        res_dict = {
            "model": model_name,
            "feature_set": "Combined (HOG + Color)",
            "num_features": 5860,
            "accuracy": round(float(acc), 4),
            "precision_fire": round(float(prec), 4),
            "recall_fire": round(float(rec), 4),
            "f1_fire": round(float(f1), 4),
            "specificity": round(float(spec), 4),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "train_time": round(float(train_time), 4),
            "prediction_time": round(float(predict_time), 4),
            "dataset_status": "REAL_DATASET_OFFICIAL"
        }
        results.append(res_dict)
        
        logger.info(f"✓ {model_name} OK | Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, Spec: {spec:.4f}")
        logger.info(f"  Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn} | Train: {train_time:.2f}s, Predict: {predict_time:.2f}s")
        
    # 4. Lưu kết quả ra file reports/results/model_comparison_OFFICIAL.csv
    fieldnames = [
        "model", "feature_set", "num_features",
        "accuracy", "precision_fire", "recall_fire", "f1_fire", "specificity",
        "tp", "tn", "fp", "fn",
        "train_time", "prediction_time", "dataset_status"
    ]
    
    with open(OFFICIAL_RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    logger.info(f"\nĐã xuất file kết quả chính thức: {OFFICIAL_RESULTS_PATH}")
    
    # 5. Hiển thị Bảng tổng hợp & Xếp hạng Hold-out
    print_official_summary(results)
    
    return results


def print_official_summary(results):
    """In bảng kết quả chính thức và xếp hạng Hold-out."""
    print("\n" + "=" * 125)
    print("BẢNG KẾT QUẢ SO SÁNH OFFICIAL BASELINE MODEL (DATASET THẬT CHÍNH THỨC)")
    print("=" * 125)
    print(f"{'Mô hình (Model)':<22} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'Specificity':<11} | {'TP':<5} | {'TN':<5} | {'FP':<5} | {'FN':<5} | {'Train(s)':<8} | {'Predict(s)':<10}")
    print("-" * 140)
    for r in results:
        print(f"{r['model']:<22} | {r['accuracy']:<8.4f} | {r['precision_fire']:<9.4f} | {r['recall_fire']:<8.4f} | {r['f1_fire']:<8.4f} | {r['specificity']:<11.4f} | {r['tp']:<5} | {r['tn']:<5} | {r['fp']:<5} | {r['fn']:<5} | {r['train_time']:<8.2f} | {r['prediction_time']:<10.2f}")
    print("=" * 125)
    
    # Tìm model tốt nhất theo các tiêu chí Hold-out
    best_acc_model = max(results, key=lambda x: x["accuracy"])
    best_rec_model = max(results, key=lambda x: x["recall_fire"])
    best_f1_model = max(results, key=lambda x: x["f1_fire"])
    
    print("\nXẾP HẠNG TẠM THỜI (OFFICIAL HOLD-OUT RESULT):")
    print(f"  🏆 Mô hình có Accuracy cao nhất:    {best_acc_model['model']} ({best_acc_model['accuracy']:.4f})")
    print(f"  🔥 Mô hình có Recall-fire cao nhất: {best_rec_model['model']} ({best_rec_model['recall_fire']:.4f})")
    print(f"  ⭐ Mô hình có F1-fire cao nhất:     {best_f1_model['model']} ({best_f1_model['f1_fire']:.4f})")
    print("\nLưu ý: Đây là xếp hạng Hold-out ban đầu. Chưa kết luận Best Model cuối cùng cho đến khi thực hiện 5-Fold Cross-Validation và Feature Comparison trên dataset thật.")
    print("=" * 125 + "\n")


if __name__ == "__main__":
    run_official_baseline_comparison()
