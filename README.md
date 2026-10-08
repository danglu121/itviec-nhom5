# GUI Project 2: Gợi ý công ty tương tự và dự đoán Recommend (ITViec)

**Đồ án tốt nghiệp Data Science, Nhóm 5**
Lư Trần Nguyên Đăng · Nguyễn Anh Duy · Phạm Thị Phương Thúy
*Trung Tâm Tin Học, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM*

Ứng dụng Streamlit cho Project 2: **Content-Based Company Similarity Recommendation
and "Recommend or Not" Classification for Candidates**.

---

## 1. Ứng dụng dành cho ai

Người dùng chọn vai khi vào app, và các mục sẽ đổi theo vai đó.

| Vai | Mục chính | Nội dung |
|---|---|---|
| **Tôi đang tìm việc** | Tìm công ty phù hợp | Bắt đầu từ một công ty đã thích, hoặc tả bằng lời nơi muốn làm việc |
| | Đọc hiểu một đánh giá | Dán một review tìm được, xem nó nghiêng về khen hay chê |
| **Tôi là nhà tuyển dụng** | Công ty cùng nhóm với chúng tôi | Xem nhóm công ty đang cạnh tranh thu hút cùng tệp ứng viên |
| | Phân tích phản hồi nhân viên | Dán phản hồi của nhân viên để phát hiện sớm dấu hiệu không hài lòng |

Hai mục còn lại dùng chung cho cả hai vai:

| Mục | Nội dung |
|---|---|
| **Toàn cảnh thị trường IT** | Lĩnh vực nào nhiều công ty, quy mô đội ngũ, nhân viên hài lòng về điều gì, và chuyện trộn hai ngôn ngữ |
| **Về ứng dụng** | Viết cho người thường đọc: số liệu đến từ đâu, chính xác tới đâu, khi nào không nên tin. Phần kỹ thuật đặt trong mục mở rộng cho ai quan tâm |

Toàn bộ nội dung viết theo hướng **giải quyết việc của người dùng**, không trình
bày quy trình xây dựng mô hình.

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
│   ├── map_2d.npy              toạ độ t-SNE để vẽ bản đồ công ty
│   └── metadata.json           chỉ số đánh giá
├── figures/
│   ├── banner.png              ảnh đầu trang, tự dựng bằng matplotlib
│   └── 01..15_*.png            14 biểu đồ phân tích từ notebook
├── requirements.txt
├── setup.sh
├── Procfile
├── runtime.txt
└── .streamlit/config.toml
```

Toàn bộ artifact chỉ **1,2 MB**, biểu đồ thêm khoảng 0,9 MB.

### Bản đồ công ty hai chiều

Trang gợi ý có một bản đồ đặt toàn bộ 477 công ty lên mặt phẳng, tô đậm công ty gốc
và các gợi ý. Toạ độ do **t-SNE** sinh ra từ 50 thành phần SVD đầu của ma trận TF-IDF.

Phép chiếu từ hơn bốn nghìn chiều xuống hai chiều luôn mất mát, nên bản đồ là hình
minh hoạ chứ không phải thước đo khoảng cách chính xác. Để biết nó mất mát tới mức
nào, tôi đã đo: **46,7% láng giềng thật rơi vào 20 điểm gần nhất trên bản đồ**, so với
4,2% nếu các điểm được đặt ngẫu nhiên. Chiếu thẳng bằng SVD hai chiều chỉ đạt 14,8%,
nên t-SNE được chọn.

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

## Minh hoạ trong ứng dụng

Ảnh đầu trang và các thẻ chọn vai (toà nhà văn phòng, cặp tài liệu, nhân viên) đều do nhóm **tự vẽ bằng matplotlib**,
mã nguồn nằm trong `draw_art.py`. Không dùng ảnh hay logo của bên thứ ba, nên
không vướng vấn đề bản quyền khi công khai repository.

---
