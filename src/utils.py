"""
utils.py - Các hàm tiện ích dùng chung cho toàn bộ project.

Bao gồm:
- Cấu hình đường dẫn
- Hằng số dùng chung
- Hàm hỗ trợ I/O
"""

import os
import logging
import cv2
import numpy as np

# ============================================================
# ĐƯỜNG DẪN GỐC CỦA PROJECT
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR         = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR        = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR        = os.path.join(PROJECT_ROOT, "reports", "figures")
RESULTS_DIR        = os.path.join(PROJECT_ROOT, "reports", "results")
ERRORS_DIR         = os.path.join(PROJECT_ROOT, "reports", "errors")

# ============================================================
# HẰNG SỐ CẤU HÌNH
# ============================================================
IMG_SIZE     = (128, 128)               # Kích thước resize ảnh (width, height)
RANDOM_SEED  = 42                       # Seed cho tái hiện kết quả
CLASS_NAMES  = ["non_fire", "fire"]     # Index 0 = non_fire, Index 1 = fire

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ============================================================
# LOGGING
# ============================================================
def setup_logger(name="datamining", level=logging.INFO):
    """Tạo logger với format chuẩn."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ============================================================
# HÀM HỖ TRỢ
# ============================================================
def ensure_dir(path):
    """Tạo thư mục nếu chưa tồn tại."""
    os.makedirs(path, exist_ok=True)


def is_image_file(filename):
    """Kiểm tra file có phải ảnh hỗ trợ không."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def imread_unicode(filepath):
    """
    Đọc ảnh hỗ trợ đường dẫn Unicode (tiếng Việt, CJK, v.v.).

    cv2.imread() trên Windows không đọc được path có ký tự Unicode.
    Workaround: đọc bytes bằng numpy.fromfile() rồi decode bằng cv2.imdecode().

    Returns:
        image: numpy array (BGR) hoặc None nếu lỗi
    """
    try:
        data = np.fromfile(filepath, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None
