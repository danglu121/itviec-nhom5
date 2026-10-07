# GUI Project 2: Gợi ý công ty tương tự và dự đoán Recommend (ITViec)

**Đồ án tốt nghiệp Data Science, Nhóm 5**
Lư Trần Nguyên Đăng · Nguyễn Anh Duy · Phạm Thị Phương Thúy
*Trung Tâm Tin Học, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM*

Ứng dụng Streamlit cho Project 2: **Content-Based Company Similarity Recommendation
and "Recommend or Not" Classification for Candidates**.

---

## 1. Ba chức năng

| Trang | Nội dung |
|---|---|
| **Gợi ý công ty tương tự** | Chọn một trong 477 công ty để xem 20 công ty gần nhất, hoặc nhập mô tả tự do bằng tiếng Việt hay tiếng Anh để tìm công ty phù hợp |
| **Dự đoán Recommend or Not** | Nhập nội dung đánh giá và 6 điểm số thành phần, nhận xác suất nhân viên sẽ giới thiệu công ty, kèm phân tích độ nhạy theo điểm số |
| **Hiệu năng mô hình** | Precision@5 so với mốc ngẫu nhiên, ma trận nhầm lẫn, và các hạn chế đã biết |

## 2. Mô hình đứng sau

| | Thuật toán | Kết quả |
|---|---|---|
| Yêu cầu 1 | TF-IDF của Gensim với `SparseMatrixSimilarity` | Precision@5 lĩnh vực **22,8%** so với mốc ngẫu nhiên 12,8% |
| Yêu cầu 2 | XGBoost trên TF-IDF 1 tới 2 từ kết hợp 6 cột điểm số | **F1 lớp "No" 0,736**, ROC-AUC 0,938 |

Cả hai được chọn sau khi so sánh **5 phương pháp** cho Yêu cầu 1 và **50 kết quả** cho
Yêu cầu 2 trong notebook phân tích. Chi tiết nằm ở `DoAn_TotNghiep_ITViec_Project2.ipynb`
của thư mục nộp bài.

### Vì sao hai thẻ của trang gợi ý dùng hai phương pháp khác nhau

Thẻ **chọn công ty có sẵn** dùng kết quả của phương pháp Gensim đã **tính sẵn lúc đóng
gói**, nên ứng dụng chạy tức thì và không cần cài Gensim.

Thẻ **nhập mô tả tự do** phải vector hoá một văn bản mới ngay lúc chạy, việc mà bảng
tính sẵn không làm được, nên dùng `TfidfVectorizer` của scikit-learn với cùng tham số
lọc từ. Hai phương pháp trùng nhau 81% số gợi ý top-5.

## 3. Cấu trúc thư mục

```
GUI_Project2/
├── app.py                  ứng dụng Streamlit
├── textprep.py             tiền xử lý nhận diện ngôn ngữ, giống hệt lúc huấn luyện
├── build_artifacts.py      script sinh artifact, KHÔNG chạy khi triển khai
├── artifacts/
│   ├── companies.parquet       477 công ty, thông tin hiển thị
│   ├── neighbors_gensim.npz    20 láng giềng gần nhất mỗi công ty, tính sẵn
│   ├── search_tfidf.joblib     vectorizer phục vụ tìm kiếm tự do
│   ├── recommend_clf.joblib    bộ phân loại XGBoost
│   └── metadata.json           chỉ số đánh giá
├── requirements.txt
├── setup.sh
├── Procfile
├── runtime.txt
└── .streamlit/config.toml
```

Toàn bộ artifact chỉ **1,2 MB**.

## 4. Dữ liệu được đóng gói ở mức tối thiểu

Đề bài ghi rõ dữ liệu không được chia sẻ, nên repo này **không chứa dữ liệu gốc**.
Cụ thể là không có ba tập tin Excel và không có 8.417 review cá nhân.

Ứng dụng chỉ cần mô hình đã huấn luyện cùng bảng thông tin công ty để hiển thị. Phần
phân loại hoạt động trên văn bản do chính người dùng nhập, nên không cần kho review.

Muốn sinh lại artifact thì chạy `build_artifacts.py` trên máy có thư mục phân tích gốc:

```bash
ITVIEC_SRC=/duong/dan/toi/Project_2 python build_artifacts.py
```

## 5. Chạy tại máy

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Mở `http://localhost:8501`. Lần chạy đầu hơi chậm vì `pyvi` phải nạp mô hình tách từ.

## 6. Triển khai lên Streamlit Cloud

1. Đẩy thư mục này lên một repository GitHub.
2. Vào [share.streamlit.io](https://share.streamlit.io), đăng nhập bằng GitHub, chọn
   **New app**.
3. Chọn repository, nhánh, và đặt **Main file path** là `app.py`.
4. Mở **Advanced settings**, chọn **Python 3.12**. Bước này quan trọng vì các phiên bản
   thư viện trong `requirements.txt` được ghim theo Python 3.12.
5. Bấm **Deploy** rồi chờ khoảng 3 tới 5 phút cho lần dựng đầu tiên.

`setup.sh` và `Procfile` có mặt theo yêu cầu của đề bài. Streamlit Cloud không dùng tới
hai tập tin này, chúng dành cho Heroku hoặc các nền tảng tương tự.

### Nếu bước dựng thất bại

Nguyên nhân hay gặp nhất là xung đột phiên bản. Khi đó hãy nới lỏng `requirements.txt`
bằng cách bỏ số phiên bản, giữ lại đúng tên thư viện:

```
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
pyarrow
altair
pyvi
```

Lưu ý rằng `recommend_clf.joblib` được tạo bằng scikit-learn 1.9 và xgboost 3.4. Nếu
Streamlit Cloud cài phiên bản khác hẳn thì có thể hiện cảnh báo lúc nạp mô hình. Trường
hợp đó cần chạy lại `build_artifacts.py` với đúng phiên bản thư viện của môi trường đích.

---

> **LƯU Ý PHÁP LÝ.** TẤT CẢ DỮ LIỆU ĐƯỢC CUNG CẤP CHỈ DÀNH CHO VIỆC HỌC TẬP, KHÔNG
> CHIA SẺ, SỬ DỤNG VỚI MỤC ĐÍCH KHÁC. Vì việc chia sẻ, sử dụng cho các mục đích khác
> đều có thể vi phạm Bản quyền và quyền sở hữu trí tuệ, Điều khoản sử dụng của trang
> web, Bảo vệ dữ liệu cá nhân... → vấn đề pháp lý.
