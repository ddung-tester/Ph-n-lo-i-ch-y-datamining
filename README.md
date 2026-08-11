# Phân loại hình ảnh cháy dựa trên đặc trưng HOG và Color Histogram

**Môn học:** Khai phá dữ liệu (Data Mining)  
**Đề tài:** So sánh các thuật toán học máy trong phân loại hình ảnh cháy dựa trên đặc trưng HOG và Color Histogram  
**Thành viên thực hiện:**
- **Trần Đình Dũng** — MSSV: B25CHHT089 (Trưởng nhóm — Technical Lead)
- **Phạm Minh Hiếu** — MSSV: B25CHHT095
- **Nguyễn Quốc Việt** — MSSV: B25CHHT121

---

## Thành viên & Phân công công việc

| STT | Họ và tên | Mã sinh viên | Vai trò đảm nhiệm | Chi tiết công việc thực hiện |
|:---:|---|:---:|---|---|
| 1 | **Trần Đình Dũng** | **B25CHHT089** | Trưởng nhóm / Technical Lead & Model Engineering | • Định hướng phương pháp luận, thiết kế kiến trúc hệ thống & pipeline dữ liệu.<br>• Lập trình module trích xuất đặc trưng HOG & Color Histogram và hệ thống Feature Cache (`.npy`).<br>• Xây dựng, huấn luyện & tối ưu hóa 4 mô hình: Logistic Regression, KNN, SVM, Random Forest.<br>• Phụ trách đánh giá chuyên sâu: 5-Fold Cross-Validation, kiểm định McNemar, PCA & ROC-AUC.<br>• Quản lý và phát triển toàn bộ bộ mã nguồn trong thư mục `src/`. |
| 2 | **Phạm Minh Hiếu** | B25CHHT095 | Data Engineering & Error Analysis | • Tiếp nhận dữ liệu công khai từ Kaggle, kiểm tra chất lượng, tổ chức cấu trúc train/test & tiền xử lý dữ liệu ảnh gốc (`data/raw`).<br>• Phân tích chi tiết danh sách các mẫu bị phân loại sai (False Positives & False Negatives).<br>• Trực quan hóa dữ liệu, vẽ biểu đồ Confusion Matrix và so sánh các bộ đặc trưng.<br>• Tham gia đóng góp và hoàn thiện tài liệu báo cáo thực nghiệm đề tài. |
| 3 | **Nguyễn Quốc Việt** | B25CHHT121 | Model Validation & Documentation | • Phụ trách kiểm thử độc lập, chạy lại pipeline thực nghiệm trên môi trường local.<br>• Thống kê, tổng hợp các bảng chỉ số hiệu năng (Accuracy, Precision, Recall, F1, AUC).<br>• Biên soạn tài liệu kỹ thuật, hướng dẫn cài đặt, cấu hình môi trường và tái hiện kết quả.<br>• Tổng hợp và chuẩn bị tài liệu, slide báo cáo trình bày đề tài. |

---

## Mô tả đề tài

Bài tập lớn thực hiện bài toán phân loại nhị phân ảnh có cháy / không cháy bằng các phương pháp học máy truyền thống. Thay vì dùng mạng nơ-ron sâu, đề tài tập trung vào việc trích xuất đặc trưng thủ công — HOG (Histogram of Oriented Gradients) và Color Histogram trong không gian màu HSV — sau đó đem so sánh hiệu năng của bốn thuật toán phân loại: Logistic Regression, K-Nearest Neighbors, SVM với kernel RBF và Random Forest.

Ngoài việc so sánh thuật toán, đề tài còn phân tích ảnh hưởng của từng bộ đặc trưng (HOG đơn, Color Histogram đơn, kết hợp cả hai) đến chất lượng phân loại.

---

## Dataset (Tập dữ liệu)

Dữ liệu gốc được nhóm sử dụng từ tập dữ liệu công khai trên Kaggle:

- **Nguồn dữ liệu gốc:** [Forest Fire and Non Fire Dataset (Kaggle)](https://www.kaggle.com/datasets/amerzishminha/forest-fire-and-non-fire-dataset)

> **Lưu ý về lưu trữ & nộp bài:**  
> Do dung lượng dữ liệu gốc lớn, repository và file nộp không bao gồm ảnh gốc. Vui lòng tải dữ liệu từ liên kết trên và đặt vào thư mục `data/raw/` theo cấu trúc hướng dẫn trong `DATASET_LINK.txt`.

Quy mô dữ liệu sau khi nhóm tổ chức và phân chia cố định:

| Tập dữ liệu | Tổng số ảnh | Fire (Cháy) | Non-fire (Không cháy) |
|---|---|---|---|
| **Train** | 15.609 | 7.804 | 7.805 |
| **Test** | 6.122 | 2.665 | 3.457 |
| **Tổng cộng** | **21.731** | **10.469** | **11.262** |

Việc phân chia tập TRAIN/TEST, tiền xử lý và trích xuất đặc trưng (HOG + Color Histogram) được thực hiện theo mã nguồn và hướng dẫn trong `README.md`.

---

## Đặc trưng sử dụng

Mỗi ảnh được resize về 128×128 và qua GaussianBlur nhẹ trước khi trích xuất.

| Bộ đặc trưng | Chiều | Mô tả |
|---|---|---|
| HOG | 1.764 | Histogram of Oriented Gradients, cell 16×16, block 2×2, 9 hướng |
| Color Histogram | 4.096 | Histogram 3D trong không gian HSV, 16 bin mỗi kênh |
| Combined | 5.860 | Ghép nối HOG + Color Histogram |

Trong tất cả thực nghiệm, `StandardScaler` chỉ được fit trên tập train, sau đó transform lên cả train lẫn test — không để rò rỉ dữ liệu.

---

## Kết quả thực nghiệm

### 1. So sánh bốn thuật toán (đặc trưng Combined, Hold-out trên TEST)

| Thuật toán | Accuracy | Precision | Recall | F1 | Specificity | ROC-AUC | Thời gian train |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 94,46% | 93,49% | 93,81% | 93,65% | 94,97% | 0,9750 | 7,5s |
| K-Nearest Neighbors | 88,29% | 90,96% | 81,16% | 85,78% | 93,78% | 0,9490 | 1,5s |
| **SVM (RBF)** | **96,64%** | **96,45%** | **95,80%** | **96,12%** | **97,28%** | **0,9929** | 467s |
| Random Forest | 96,52% | 96,12% | 95,87% | 96,00% | 97,02% | 0,9924 | 50s |

### 2. So sánh bộ đặc trưng (SVM RBF, Hold-out trên TEST)

| Bộ đặc trưng | Chiều | Accuracy | Recall | F1 |
|---|---|---|---|---|
| HOG | 1.764 | 94,58% | 93,58% | 93,76% |
| Color Histogram | 4.096 | 95,20% | 94,15% | 94,47% |
| **Combined** | **5.860** | **96,64%** | **95,80%** | **96,12%** |

Color Histogram đơn cho kết quả tốt hơn HOG đơn. Kết hợp cả hai đặc trưng luôn cho kết quả cao nhất ở mọi chỉ số.

### 3. 5-Fold Stratified Cross-Validation trên TRAIN

| Thuật toán | CV Accuracy (mean ± std) | CV Recall (mean ± std) | CV F1 (mean ± std) |
|---|---|---|---|
| Logistic Regression | 94,50% ± 0,45% | 94,21% ± 0,63% | 94,49% ± 0,46% |
| K-Nearest Neighbors | 88,09% ± 0,52% | 81,18% ± 0,87% | 87,20% ± 0,59% |
| **SVM (RBF)** | **96,25% ± 0,53%** | 95,14% ± 0,60% | **96,21% ± 0,54%** |
| Random Forest | 96,21% ± 0,37% | **95,89% ± 0,53%** | 96,20% ± 0,37% |

Kết quả CV bám sát Hold-out — cho thấy mô hình ổn định, không overfit.

### 4. Kiểm định thống kê McNemar (SVM vs. Random Forest)

| Chỉ số | Giá trị |
|---|---|
| Cả hai đúng | 5.804 |
| SVM đúng / RF sai | 112 |
| SVM sai / RF đúng | 105 |
| Cả hai sai | 101 |
| χ² | 0,1659 |
| p-value | 0,6838 |

Với p = 0,6838 > α = 0,05, **không có sự khác biệt có ý nghĩa thống kê** giữa SVM và Random Forest trên tập TEST.

---

## Cấu trúc thư mục

```
phan-loai-chay-datamining/
│
├── data/
│   ├── raw/                ← Ảnh gốc (Tải từ Kaggle theo DATASET_LINK.txt)
│   │   ├── train/          ← Ảnh gốc tập train (fire/ + non_fire/)
│   │   └── test/           ← Ảnh gốc tập test  (fire/ + non_fire/)
│   └── processed/
│       ├── train/          ← Feature cache: X_hog.npy, X_color.npy, X_combined.npy, y.npy
│       ├── test/           ← Feature cache TEST
│       └── metadata.csv    ← Thống kê số lượng mẫu
│
├── models/
│   └── official/
│       ├── logistic_regression_combined.pkl  ← Model đã train (nhỏ, được push)
│       ├── svm_combined.pkl                  ← (nặng ~240MB, không push)
│       ├── knn_combined.pkl                  ← (nặng ~349MB, không push)
│       └── random_forest_combined.pkl        ← (nặng ~18MB, không push)
│
├── reports/
│   ├── figures/            ← Biểu đồ so sánh, confusion matrix, ROC curve, PCA
│   ├── results/            ← Kết quả CSV chính thức (*_OFFICIAL.csv)
│   └── errors/             ← Ảnh phân loại sai (False Positive / False Negative)
│
├── src/
│   ├── utils.py                        ← Cấu hình đường dẫn, hằng số, hàm dùng chung
│   ├── data_loader.py                  ← Đọc ảnh từ dataset
│   ├── preprocess.py                   ← Tiền xử lý (resize, denoise)
│   ├── features.py                     ← Trích xuất HOG + Color Histogram
│   ├── feature_cache.py                ← Xây dựng và load feature cache
│   ├── train_compare.py                ← Huấn luyện + Hold-out 4 thuật toán
│   ├── train_cv_official.py            ← 5-Fold Stratified Cross-Validation
│   ├── feature_compare_official.py     ← So sánh 3 bộ đặc trưng (SVM)
│   ├── final_evaluation_official.py    ← Confusion Matrix + Error Analysis
│   └── advanced_analysis_official.py   ← ROC-AUC + PCA + McNemar Test
│
├── requirements.txt
├── DATASET_LINK.txt                    ← Link & hướng dẫn tải dữ liệu Kaggle
├── GITHUB_LINK.txt                     ← Link GitHub repository của dự án
└── README.md
```

---

## Quy chuẩn nộp bài (File ZIP)

File nộp bài chính thức của nhóm (`Nhom_Dung_Hieu_Viet_DataMining.zip`) bao gồm các thành phần sau:

```text
Nhom_Dung_Hieu_Viet_DataMining.zip
├── src/
├── reports/
│   ├── results/
│   └── figures/
├── requirements.txt
├── README.md
├── DATASET_LINK.txt
└── GITHUB_LINK.txt
```

**Các mục KHÔNG bao gồm trong file nộp (để tối ưu dung lượng):**
- `data/raw/`: Nguyên nhân chính khiến dung lượng vượt 10GB (ảnh gốc tải về trực tiếp từ Kaggle theo `DATASET_LINK.txt`).
- File cache đặc trưng `.npy`.
- Các file mô hình `.pkl` dung lượng lớn.
- Thư mục môi trường ảo `.venv/`, `.git/`, `__pycache__/`.

---

## Hướng dẫn chạy lại thực nghiệm

### Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

**Yêu cầu:** Python 3.10+

### Các bước thực nghiệm theo thứ tự

```bash
# Bước 1 — Xây dựng feature cache (cần có ảnh trong data/raw/)
python -m src.feature_cache

# Bước 2 — Huấn luyện 4 mô hình + đánh giá Hold-out
python -m src.train_compare

# Bước 3 — 5-Fold Cross-Validation
python -m src.train_cv_official

# Bước 4 — So sánh bộ đặc trưng (HOG / Color Hist / Combined)
python -m src.feature_compare_official

# Bước 5 — Confusion Matrix + Error Analysis
python -m src.final_evaluation_official

# Bước 6 — ROC-AUC + PCA + McNemar Test
python -m src.advanced_analysis_official
```

> **Lưu ý:** Bước 1 cần có ảnh gốc trong `data/raw/train/` và `data/raw/test/` (tải về từ [Kaggle](https://www.kaggle.com/datasets/amerzishminha/forest-fire-and-non-fire-dataset)). Các Bước 2–6 có thể chạy trực tiếp bằng feature cache `.npy` trong `data/processed/` mà không cần ảnh gốc.

---

## Thư viện sử dụng

| Thư viện | Phiên bản | Mục đích |
|---|---|---|
| scikit-learn | ≥ 1.3 | Thuật toán phân loại, CV, metrics |
| scikit-image | ≥ 0.21 | Trích xuất HOG |
| opencv-python | ≥ 4.8 | Đọc ảnh, Color Histogram, resize |
| numpy | ≥ 1.24 | Xử lý mảng số |
| matplotlib / seaborn | ≥ 3.7 / 0.12 | Vẽ biểu đồ |
| scipy | ≥ 1.11 | McNemar Test |
| pandas | ≥ 2.0 | Xử lý CSV |

---

## Kết quả lưu tại

| File | Nội dung |
|---|---|
| `reports/results/model_comparison_OFFICIAL.csv` | Hold-out: Accuracy, F1, Recall, Specificity, AUC, thời gian train 4 mô hình |
| `reports/results/cross_validation_OFFICIAL.csv` | CV 5-fold: mean ± std của 4 mô hình |
| `reports/results/feature_comparison_OFFICIAL.csv` | So sánh HOG / Color Hist / Combined |
| `reports/results/final_summary_OFFICIAL.csv` | Tổng hợp Hold-out + CV |
| `reports/results/roc_auc_OFFICIAL.csv` | ROC-AUC 4 mô hình |
| `reports/results/mcnemar_svm_vs_rf_OFFICIAL.csv` | Kết quả McNemar Test |
| `reports/results/pca_summary_OFFICIAL.csv` | Explained variance PCA 2D |
| `reports/figures/roc_curve_OFFICIAL.png` | Đường cong ROC 4 mô hình |
| `reports/figures/pca_feature_space_OFFICIAL.png` | Phân bố dữ liệu trong không gian PCA 2D |
| `reports/figures/cm_*_OFFICIAL.png` | Confusion Matrix từng mô hình |
| `reports/errors/svm/`, `reports/errors/random_forest/` | Ảnh phân loại sai (FP và FN) |