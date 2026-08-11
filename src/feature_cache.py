"""
feature_cache.py - Trích xuất và lưu trữ cache đặc trưng (HOG, Color Histogram, Combined)
cho DATASET THẬT CHÍNH THỨC (Forest Fire & Non Fire Dataset) với tối ưu Multiprocessing.

Cấu trúc lưu trữ trong data/processed/:
data/processed/
├── train/
│   ├── X_hog.npy
│   ├── X_color.npy
│   ├── X_combined.npy
│   ├── y.npy
│   └── metadata.csv
└── test/
    ├── X_hog.npy
    ├── X_color.npy
    ├── X_combined.npy
    ├── y.npy
    └── metadata.csv

Định dạng kiểu dữ liệu:
- Feature arrays được lưu dạng np.float32 để tiết kiệm RAM và dung lượng đĩa cho 21.731 mẫu.
- y được lưu dạng np.int64.
"""

import os
import sys
import csv
import argparse
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Reconfigure stdout cho Windows console để hiển thị UTF-8 không bị lỗi
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Thêm PROJECT_ROOT vào sys.path để chạy trực tiếp script không bị lỗi import
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_single_image
from src.preprocess import preprocess_single
from src.features import extract_hog, extract_color_histogram, get_feature_info
from src.utils import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, ensure_dir, setup_logger, is_image_file
)

logger = setup_logger("feature_cache")

# Đường dẫn cache theo từng tập
TRAIN_PROCESSED_DIR = os.path.join(DATA_PROCESSED_DIR, "train")
TEST_PROCESSED_DIR = os.path.join(DATA_PROCESSED_DIR, "test")

# Mapping label chuẩn hóa
LABEL_MAP = {
    "fire": 1,
    "non fire": 0,
    "non_fire": 0
}

CLASS_NAME_MAP = {
    1: "fire",
    0: "non_fire"
}


def _process_single_image_worker(args_tuple):
    """
    Worker function chạy song song trên các CPU core.
    """
    path, label, split_name = args_tuple
    filename = os.path.basename(path)
    class_name = CLASS_NAME_MAP[label]
    
    img = load_single_image(path)
    if img is None:
        return None
        
    try:
        # Preprocess 100% giữ nguyên pipeline: Resize 128x128 + GaussianBlur
        processed_img = preprocess_single(img)
        
        # Trích xuất đặc trưng
        hog_feat = extract_hog(processed_img).astype(np.float32)
        color_feat = extract_color_histogram(processed_img).astype(np.float32)
        combined_feat = np.concatenate([hog_feat, color_feat]).astype(np.float32)
        
        meta = {
            "split": split_name.upper(),
            "filename": filename,
            "image_path": path,
            "class_name": class_name,
            "label": label
        }
        
        return (hog_feat, color_feat, combined_feat, label, meta)
    except Exception:
        return None


def get_split_paths(split="train"):
    """Trả về dict đường dẫn các tệp cache cho tập split ('train' hoặc 'test')."""
    target_dir = TRAIN_PROCESSED_DIR if split.lower() == "train" else TEST_PROCESSED_DIR
    return {
        "dir": target_dir,
        "hog": os.path.join(target_dir, "X_hog.npy"),
        "color": os.path.join(target_dir, "X_color.npy"),
        "combined": os.path.join(target_dir, "X_combined.npy"),
        "y": os.path.join(target_dir, "y.npy"),
        "metadata": os.path.join(target_dir, "metadata.csv")
    }


def check_cache_exists(split="train"):
    """Kiểm tra xem tất cả tệp cache của một tập split đã tồn tại đầy đủ chưa."""
    paths = get_split_paths(split)
    required_files = [paths["hog"], paths["color"], paths["combined"], paths["y"], paths["metadata"]]
    return all(os.path.isfile(f) for f in required_files)


def load_cache(split="train"):
    """
    Tải dữ liệu đặc trưng từ cache trong data/processed/train/ hoặc data/processed/test/.
    
    Parameters:
        split: "train" hoặc "test"
        
    Returns:
        X_hog (float32), X_color (float32), X_combined (float32), y (int64)
    """
    split = split.lower()
    if not check_cache_exists(split):
        raise FileNotFoundError(f"Chưa có đầy đủ tệp cache cho tập '{split}' trong data/processed/{split}/.")
    
    paths = get_split_paths(split)
    logger.info(f"Đang tải feature cache tập [{split.upper()}] từ: {paths['dir']}")
    
    X_hog = np.load(paths["hog"])
    X_color = np.load(paths["color"])
    X_combined = np.load(paths["combined"])
    y = np.load(paths["y"])
    
    return X_hog, X_color, X_combined, y


def load_image_paths_for_split(split="train"):
    """
    Quét thư mục data/raw/train/ hoặc data/raw/test/ để lấy danh sách ảnh và nhãn.
    Hỗ trợ cả tên thư mục 'non fire' và 'non_fire'.
    """
    split_dir = os.path.join(DATA_RAW_DIR, split.lower())
    if not os.path.isdir(split_dir):
        alt_dir = os.path.join(DATA_RAW_DIR, "archive", "FOREST_FIRE_DATASET", split.lower())
        if os.path.isdir(alt_dir):
            split_dir = alt_dir
        else:
            logger.error(f"Không tìm thấy thư mục raw cho tập '{split}': {split_dir}")
            return [], []
            
    image_paths = []
    labels = []
    
    for sub_name in sorted(os.listdir(split_dir)):
        sub_dir = os.path.join(split_dir, sub_name)
        if not os.path.isdir(sub_dir):
            continue
            
        norm_key = sub_name.lower().strip()
        if norm_key not in LABEL_MAP:
            logger.warning(f"Bỏ qua thư mục không xác định trong {split}: {sub_name}")
            continue
            
        label = LABEL_MAP[norm_key]
        count = 0
        
        for filename in sorted(os.listdir(sub_dir)):
            if is_image_file(filename):
                filepath = os.path.join(sub_dir, filename)
                image_paths.append(filepath)
                labels.append(label)
                count += 1
                
        canonical_class = CLASS_NAME_MAP[label]
        logger.info(f"  [{split.upper()} / {canonical_class}] Tìm thấy {count} ảnh (folder gốc: '{sub_name}')")
        
    logger.info(f"Tổng tập [{split.upper()}]: {len(image_paths)} ảnh")
    return image_paths, labels


def process_and_save_split(split="train", force_rebuild=False, max_workers=None):
    """
    Trích xuất đặc trưng đa tiến trình (multiprocessing) và lưu cache cho 1 tập split.
    """
    split = split.lower()
    paths = get_split_paths(split)
    ensure_dir(paths["dir"])
    
    if check_cache_exists(split) and not force_rebuild:
        logger.info(f"Cache tập [{split.upper()}] đã tồn tại. Sử dụng cache hiện có.")
        X_hog, X_color, X_combined, y = load_cache(split)
        validate_and_print_split_summary(split, X_hog, X_color, X_combined, y)
        return X_hog, X_color, X_combined, y

    logger.info("=" * 65)
    logger.info(f"BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG TẬP [{split.upper()}] (MULTIPROCESSING)")
    logger.info("=" * 65)
    
    start_time = time.time()
    
    image_paths, labels = load_image_paths_for_split(split)
    total_samples = len(image_paths)
    
    if total_samples == 0:
        logger.error(f"Không tìm thấy ảnh nào trong data/raw/{split}!")
        sys.exit(1)
        
    feat_info = get_feature_info()
    expected_hog_dim = feat_info["hog_size"]
    expected_color_dim = feat_info["color_hist_size"]
    expected_combined_dim = feat_info["total_features"]
    
    tasks = [(p, l, split) for p, l in zip(image_paths, labels)]
    
    hog_list = []
    color_list = []
    combined_list = []
    valid_labels = []
    metadata_rows = []
    
    success_count = 0
    error_count = 0
    
    # Sử dụng ProcessPoolExecutor để tận dụng tối đa tất cả các nhân CPU
    cpu_count = os.cpu_count() or 4
    workers = max_workers or min(cpu_count, 16)
    logger.info(f"Sử dụng ProcessPoolExecutor với {workers} CPU workers...")
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit tất cả các công việc theo thứ tự
        results_map = executor.map(_process_single_image_worker, tasks, chunksize=64)
        
        for idx, res in enumerate(results_map, 1):
            if res is None:
                error_count += 1
                continue
                
            hog_feat, color_feat, combined_feat, label, meta = res
            
            assert hog_feat.shape[0] == expected_hog_dim, f"Lỗi chiều HOG: {hog_feat.shape[0]}"
            assert color_feat.shape[0] == expected_color_dim, f"Lỗi chiều Color: {color_feat.shape[0]}"
            assert combined_feat.shape[0] == expected_combined_dim, f"Lỗi chiều Combined: {combined_feat.shape[0]}"
            
            hog_list.append(hog_feat)
            color_list.append(color_feat)
            combined_list.append(combined_feat)
            valid_labels.append(label)
            metadata_rows.append(meta)
            success_count += 1
            
            if idx % 2000 == 0 or idx == total_samples:
                pct = (idx / total_samples) * 100
                logger.info(f"Tiến độ [{split.upper()}]: {idx}/{total_samples} ({pct:.1f}%) - Thành công: {success_count}, Lỗi: {error_count}")
                
    # Chuyển đổi mảng numpy (float32 cho feature, int64 cho label)
    X_hog = np.array(hog_list, dtype=np.float32)
    X_color = np.array(color_list, dtype=np.float32)
    X_combined = np.array(combined_list, dtype=np.float32)
    y = np.array(valid_labels, dtype=np.int64)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Hoàn thành trích xuất tập [{split.upper()}] trong {elapsed_time:.2f} giây.")
    
    # Lưu file npy & csv
    logger.info(f"Đang lưu trữ cache tập [{split.upper()}] vào {paths['dir']}...")
    np.save(paths["hog"], X_hog)
    np.save(paths["color"], X_color)
    np.save(paths["combined"], X_combined)
    np.save(paths["y"], y)
    
    fieldnames = ["split", "filename", "image_path", "class_name", "label"]
    with open(paths["metadata"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)
        
    logger.info(f"Đã lưu xong toàn bộ tệp cache tập [{split.upper()}]!")
    
    validate_and_print_split_summary(split, X_hog, X_color, X_combined, y, success_count, error_count, elapsed_time)
    
    return X_hog, X_color, X_combined, y


def validate_and_print_split_summary(split, X_hog, X_color, X_combined, y, success_count=None, error_count=None, elapsed_time=None):
    """Kiểm tra tính toàn vẹn và in thống kê chi tiết cho một tập split."""
    paths = get_split_paths(split)
    n_samples = len(y)
    feat_info = get_feature_info()
    
    print("\n" + "=" * 65)
    print(f"BÁO CÁO KIỂM TRA FEATURE CACHE TẬP [{split.upper()}]")
    print("=" * 65)
    
    if success_count is not None:
        print(f"Số ảnh đọc thành công: {success_count}")
        print(f"Số ảnh bị lỗi:       {error_count}")
        print(f"Thời gian trích xuất:  {elapsed_time:.2f} giây")
        
    # Check 1: Số mẫu đồng bộ
    assert X_hog.shape[0] == n_samples, "Mismatch X_hog và y!"
    assert X_color.shape[0] == n_samples, "Mismatch X_color và y!"
    assert X_combined.shape[0] == n_samples, "Mismatch X_combined và y!"
    print(f"✓ Kiểm tra số mẫu: Đồng bộ {n_samples} mẫu dữ liệu.")
    
    # Check 2: Kiểu dữ liệu và số chiều
    assert X_hog.dtype == np.float32, f"X_hog dtype {X_hog.dtype} != float32"
    assert X_color.dtype == np.float32, f"X_color dtype {X_color.dtype} != float32"
    assert X_combined.dtype == np.float32, f"X_combined dtype {X_combined.dtype} != float32"
    
    assert X_hog.shape[1] == 1764, f"X_hog dim {X_hog.shape[1]} != 1764"
    assert X_color.shape[1] == 4096, f"X_color dim {X_color.shape[1]} != 4096"
    assert X_combined.shape[1] == 5860, f"X_combined dim {X_combined.shape[1]} != 5860"
    assert X_combined.shape[1] == X_hog.shape[1] + X_color.shape[1], "Combined dim != HOG + Color!"
    
    print(f"✓ Kiểm tra kiểu dữ liệu & số chiều (np.float32):")
    print(f"  - X_hog.shape:      {X_hog.shape}")
    print(f"  - X_color.shape:    {X_color.shape}")
    print(f"  - X_combined.shape: {X_combined.shape} (1764 + 4096 = 5860)")
    
    # Check 3: NaN và Inf
    assert not np.isnan(X_hog).any(), f"NaN trong X_hog tập {split}!"
    assert not np.isnan(X_color).any(), f"NaN trong X_color tập {split}!"
    assert not np.isnan(X_combined).any(), f"NaN trong X_combined tập {split}!"
    assert not np.isinf(X_hog).any(), f"Inf trong X_hog tập {split}!"
    assert not np.isinf(X_color).any(), f"Inf trong X_color tập {split}!"
    assert not np.isinf(X_combined).any(), f"Inf trong X_combined tập {split}!"
    print("✓ Kiểm tra NaN / Inf: Không có NaN/Inf.")
    
    # Check 4: Phân bố class
    fire_c = int(np.sum(y == 1))
    non_fire_c = int(np.sum(y == 0))
    print(f"Phân bố nhãn tập [{split.upper()}]:")
    print(f"  - 🔥 fire (label 1):     {fire_c} ảnh ({fire_c/n_samples*100:.2f}%)")
    print(f"  - 🌲 non_fire (label 0): {non_fire_c} ảnh ({non_fire_c/n_samples*100:.2f}%)")
    
    # Check 5: Dung lượng đĩa
    files = [("X_hog.npy", paths["hog"]), ("X_color.npy", paths["color"]),
             ("X_combined.npy", paths["combined"]), ("y.npy", paths["y"]),
             ("metadata.csv", paths["metadata"])]
    print(f"Dung lượng tệp cache tập [{split.upper()}]:")
    for name, fpath in files:
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  - {name:<16}: {size_mb:.2f} MB")
    print("=" * 65)


def build_feature_cache(force_rebuild=False):
    """
    Xây dựng cache đặc trưng tách biệt cho cả tập TRAIN và TEST.
    """
    print("\n" + "=" * 70)
    print("XÂY DỰNG FEATURE CACHE CHO DATASET CHÍNH THỨC (TRAIN VÀ TEST TÁCH BIỆT)")
    print("=" * 70)
    
    t_start = time.time()
    
    # Process Train
    process_and_save_split("train", force_rebuild=force_rebuild)
    
    # Process Test
    process_and_save_split("test", force_rebuild=force_rebuild)
    
    t_total = time.time() - t_start
    logger.info(f"\n Hoàn tất xây dựng Feature Cache toàn bộ dataset trong {t_total:.2f} giây!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo cache đặc trưng HOG và Color Histogram cho dataset thật.")
    parser.add_argument("--force", action="store_true", help="Bắt buộc trích xuất và ghi đè lại cache dữ liệu.")
    args = parser.parse_args()
    
    build_feature_cache(force_rebuild=args.force)
