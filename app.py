"""Ứng dụng web ITViec: gợi ý công ty tương tự và dự đoán Recommend.

Chạy:  streamlit run app.py

Ứng dụng chỉ ĐỌC artifact đã đóng gói sẵn trong thư mục artifacts/, không
huấn luyện lại lúc chạy. Xem build_artifacts.py để biết cách tạo artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from textprep import preprocess

APP_DIR = Path(__file__).parent
ART = APP_DIR / "artifacts"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MUTED, DANGER = "#52514e", "#8f2222"

NUMERIC = ["Rating", "Salary & benefits", "Training & learning",
           "Management cares about me", "Culture & fun", "Office & workspace"]
NUMERIC_VI = {
    "Rating": "Đánh giá chung",
    "Salary & benefits": "Lương và phúc lợi",
    "Training & learning": "Đào tạo và học hỏi",
    "Management cares about me": "Quản lý quan tâm nhân viên",
    "Culture & fun": "Văn hoá và không khí",
    "Office & workspace": "Văn phòng và chỗ làm việc",
}

st.set_page_config(page_title="ITViec — Gợi ý công ty & Dự đoán Recommend",
                   page_icon="💼", layout="wide")


# ═══════════════════════════════════════════════════ Nạp artifact
@st.cache_data
def load_companies() -> pd.DataFrame:
    return pd.read_parquet(ART / "companies.parquet")


@st.cache_data
def load_neighbors() -> tuple[np.ndarray, np.ndarray]:
    z = np.load(ART / "neighbors_gensim.npz")
    return z["idx"].astype(int), z["score"]


@st.cache_resource
def load_search():
    return joblib.load(ART / "search_tfidf.joblib")


@st.cache_resource
def load_classifier():
    return joblib.load(ART / "recommend_clf.joblib")


@st.cache_data
def load_meta() -> dict:
    return json.loads((ART / "metadata.json").read_text(encoding="utf-8"))


co = load_companies()
meta = load_meta()
m2 = meta["yeucau2"]

# ═══════════════════════════════════════════════════ Thanh bên
st.sidebar.title("💼 ITViec")
st.sidebar.caption("Gợi ý công ty tương tự và dự đoán khả năng Recommend")
page = st.sidebar.radio(
    "Chức năng",
    ["Gợi ý công ty tương tự", "Dự đoán Recommend or Not", "Hiệu năng mô hình"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.caption(
    f"**Yêu cầu 1** · {meta['n_companies']} công ty\n\n"
    f"{meta['yeucau1']['method']}\n\n"
    f"**Yêu cầu 2** · {m2['n_train']:,} review huấn luyện\n\n"
    f"XGBoost · F1 lớp \"No\" {m2['F1_No']:.3f} · AUC {m2['ROC_AUC']:.3f}"
)
st.sidebar.divider()
st.sidebar.caption(
    "Đồ án tốt nghiệp Data Science, Nhóm 5\n\n"
    "Dữ liệu chỉ dùng cho mục đích học tập."
)


def ket_qua_goi_y(rows: pd.DataFrame, scores: np.ndarray, goc: pd.Series | None):
    """Hiển thị danh sách công ty được gợi ý dưới dạng thẻ."""
    for rank, ((_, r), s) in enumerate(zip(rows.iterrows(), scores), 1):
        trung = (goc is not None
                 and r["Company industry"] == goc["Company industry"])
        dau = "🟢" if trung else "⚪"
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{rank}. {r['Company Name']}**")
            c2.markdown(f"<div style='text-align:right;color:{MUTED}'>"
                        f"{s:.3f}</div>", unsafe_allow_html=True)
            st.caption(
                f"{dau} {r['Company industry']}  ·  {r['Company Type']}  ·  "
                f"{r['Company size']}  ·  {r['Country']}"
            )
            mo_ta = str(r["text_raw"])[:360]
            st.write(mo_ta + ("..." if len(str(r["text_raw"])) > 360 else ""))
            if isinstance(r["Href"], str) and r["Href"].startswith("http"):
                st.markdown(f"[Xem trên ITViec]({r['Href']})")


# ═══════════════════════════════════════ 1. Gợi ý công ty tương tự
if page == "Gợi ý công ty tương tự":
    st.title("Gợi ý công ty tương tự")
    st.caption(
        f"Content-Based Filtering trên mô tả của {meta['n_companies']} công ty. "
        "Hệ thống so sánh nội dung văn bản, không dùng lịch sử tương tác."
    )

    tab1, tab2 = st.tabs(["Chọn một công ty có sẵn", "Nhập mô tả tự do"])

    # --- Cách 1: dùng láng giềng tính sẵn bằng phương pháp B (Gensim)
    with tab1:
        ten = st.selectbox("Chọn công ty", co["Company Name"].tolist(),
                           index=int(co.index[co["Company Name"]
                                              .str.contains("FPT Software",
                                                            na=False)][:1][0])
                           if co["Company Name"].str.contains("FPT Software",
                                                              na=False).any() else 0)
        k = st.slider("Số công ty gợi ý", 3, meta["topk"], 5, key="k1")

        i = int(co.index[co["Company Name"] == ten][0])
        goc = co.loc[i]

        with st.container(border=True):
            st.markdown(f"### {goc['Company Name']}")
            st.caption(f"{goc['Company industry']}  ·  {goc['Company Type']}  ·  "
                       f"{goc['Company size']}  ·  {goc['Country']}  ·  "
                       f"Làm việc {goc['Working days']}  ·  {goc['Overtime Policy']}")
            with st.expander("Mô tả công ty"):
                st.write(goc["text_raw"])

        st.subheader(f"{k} công ty tương tự nhất")
        st.caption("Chấm xanh nghĩa là trùng lĩnh vực hoạt động với công ty gốc. "
                   "Lĩnh vực không được đưa vào mô hình, nên đây là thước đo "
                   "độc lập để bạn tự kiểm chứng chất lượng gợi ý.")

        nn_idx, nn_score = load_neighbors()
        top = nn_idx[i][:k]
        ket_qua_goi_y(co.loc[top], nn_score[i][:k], goc)

    # --- Cách 2: vector hoá văn bản mới, phải dùng sklearn lúc chạy
    with tab2:
        st.caption(
            "Nhập mô tả về công ty bạn đang tìm, bằng tiếng Việt hoặc tiếng Anh. "
            "Hệ thống tự nhận diện ngôn ngữ rồi tách từ tương ứng."
        )
        truy_van = st.text_area(
            "Mô tả công ty bạn muốn tìm",
            "Công ty fintech phát triển sản phẩm thanh toán, môi trường trẻ, "
            "có cơ hội học hỏi về blockchain và dữ liệu lớn",
            height=110,
        )
        k2 = st.slider("Số công ty gợi ý", 3, 20, 5, key="k2")

        if st.button("Tìm công ty phù hợp", type="primary",
                     use_container_width=True):
            sach, lang = preprocess(truy_van)
            if not sach.strip():
                st.warning("Văn bản sau khi làm sạch không còn từ nào. "
                           "Hãy nhập mô tả dài hơn.")
            else:
                s = load_search()
                q = s["vectorizer"].transform([sach])
                sim = cosine_similarity(q, s["matrix"]).ravel()
                top = np.argsort(-sim)[:k2]

                st.caption(
                    f"Ngôn ngữ nhận diện: **{'tiếng Việt' if lang == 'vi' else 'tiếng Anh'}**  ·  "
                    f"Số từ sau khi làm sạch: **{len(sach.split())}**"
                )
                if sim[top[0]] < 0.05:
                    st.warning(
                        "Điểm tương tự cao nhất rất thấp. Mô tả của bạn có thể "
                        "dùng từ ngữ không xuất hiện trong kho dữ liệu, "
                        "hoặc khác ngôn ngữ với phần lớn mô tả công ty."
                    )
                ket_qua_goi_y(co.loc[top], sim[top], None)

        with st.expander("Vì sao hai thẻ dùng hai phương pháp khác nhau?"):
            st.markdown(
                """
Thẻ **Chọn một công ty có sẵn** dùng **phương pháp B, TF-IDF của Gensim**. Đây là
phương pháp thắng trong phần đánh giá ở notebook. Kết quả được tính sẵn lúc đóng gói
nên ứng dụng chạy tức thì và không cần cài Gensim.

Thẻ **Nhập mô tả tự do** phải vector hoá một văn bản **mới** ngay lúc chạy, việc mà
bảng tính sẵn không làm được. Thẻ này vì vậy dùng **phương pháp A, TF-IDF của
scikit-learn**, cùng tham số lọc từ.

Hai phương pháp trùng nhau **81%** số gợi ý top-5 và chênh nhau khoảng một điểm phần
trăm về Precision@5, nên lựa chọn này không làm thay đổi chất lượng một cách đáng kể.
                """
            )

# ═══════════════════════════════════ 2. Dự đoán Recommend or Not
elif page == "Dự đoán Recommend or Not":
    st.title("Dự đoán Recommend or Not")
    st.caption(
        "Nhập nội dung đánh giá và các điểm số thành phần để dự đoán khả năng "
        "nhân viên sẽ giới thiệu công ty cho bạn bè."
    )

    trai, phai = st.columns([3, 2])

    with trai:
        st.markdown("**Nội dung đánh giá**")
        tieu_de = st.text_input("Tiêu đề", "Môi trường tốt nhưng lương chưa cạnh tranh")
        thich = st.text_area(
            "Điều bạn thích (What I liked)",
            "Đồng nghiệp thân thiện, sếp lắng nghe, được tham gia dự án lớn và "
            "học hỏi nhiều công nghệ mới. Văn phòng đẹp, giờ giấc linh hoạt.",
            height=120,
        )
        gop_y = st.text_area(
            "Đề xuất cải thiện (Suggestions for improvement)",
            "Lương chưa theo kịp thị trường, quy trình review lương không rõ ràng. "
            "Thỉnh thoảng phải tăng ca khi gần deadline.",
            height=110,
        )

    with phai:
        st.markdown("**Điểm số thành phần** (thang 1 đến 5)")
        diem = {}
        for col in NUMERIC:
            diem[col] = st.slider(NUMERIC_VI[col], 1.0, 5.0, 4.0, 0.5,
                                  key=f"sl_{col}")

    nguong = st.slider(
        "Ngưỡng quyết định cho nhãn \"Yes\"", 0.10, 0.90, 0.50, 0.05,
        help="Hạ ngưỡng để bắt được nhiều trường hợp \"No\" hơn, đổi lại số "
             "báo động giả tăng lên.",
    )

    if st.button("Dự đoán", type="primary", use_container_width=True):
        tho = f"{tieu_de} {thich} {gop_y}".strip()
        sach, lang = preprocess(tho)

        if not sach.strip():
            st.warning("Văn bản sau khi làm sạch không còn từ nào. "
                       "Hãy nhập nội dung đánh giá.")
        else:
            X = pd.DataFrame([{**diem, "text_clean": sach}])
            p_yes = float(load_classifier().predict_proba(X)[0, 1])
            nhan = "Yes" if p_yes >= nguong else "No"
            mau = SERIES[2] if nhan == "Yes" else SERIES[1]
            icon = "👍" if nhan == "Yes" else "👎"
            dien_giai = ("sẽ giới thiệu công ty cho bạn bè"
                         if nhan == "Yes" else "sẽ KHÔNG giới thiệu công ty")

            st.divider()
            st.markdown(
                f"<div style='padding:18px;border-radius:10px;background:{mau}18;"
                f"border-left:5px solid {mau}'>"
                f"<div style='font-size:26px;font-weight:700;color:{mau}'>"
                f"{icon} Recommend = {nhan}</div>"
                f"<div style='color:{MUTED};margin-top:6px'>Mô hình dự đoán người "
                f"viết đánh giá này <b>{dien_giai}</b>. Xác suất \"Yes\" "
                f"<b>{p_yes * 100:.1f}%</b>, ngưỡng quyết định "
                f"{nguong * 100:.0f}%.</div></div>",
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Xác suất Yes", f"{p_yes * 100:.1f}%")
            c2.metric("Điểm trung bình đã nhập",
                      f"{np.mean(list(diem.values())):.2f} / 5")
            c3.metric("Ngôn ngữ nhận diện",
                      "Tiếng Việt" if lang == "vi" else "Tiếng Anh")

            thanh = pd.DataFrame({
                "nhan": ["No", "Yes"],
                "p": [1 - p_yes, p_yes],
            })
            nen = alt.Chart(thanh).encode(
                y=alt.Y("nhan:N", sort=["No", "Yes"], title=None,
                        axis=alt.Axis(labelFontSize=14, labelPadding=10)),
            )
            cot = nen.mark_bar(cornerRadius=4, size=34).encode(
                x=alt.X("p:Q", scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format="%"), title="Xác suất"),
                color=alt.Color("nhan:N",
                                scale=alt.Scale(domain=["No", "Yes"],
                                                range=[SERIES[1], SERIES[2]]),
                                legend=None),
                tooltip=[alt.Tooltip("nhan:N", title="Nhãn"),
                         alt.Tooltip("p:Q", format=".1%", title="Xác suất")],
            )
            # Nhãn số đặt ngay cạnh đầu cột. Với xác suất rất nhỏ thì cột gần
            # như vô hình, lúc đó chỉ còn con số cho biết giá trị thật.
            chu = nen.mark_text(align="left", dx=7, fontSize=12, color=MUTED).encode(
                x=alt.X("p:Q", scale=alt.Scale(domain=[0, 1])),
                text=alt.Text("p:Q", format=".1%"),
            )
            st.altair_chart((cot + chu).properties(height=160),
                            use_container_width=True)

            st.caption(
                f"Văn bản sau khi làm sạch ({len(sach.split())} từ): "
                f"`{sach[:300]}{'...' if len(sach) > 300 else ''}`"
            )

            # Phân tích độ nhạy: giữ nguyên văn bản, quét điểm số từ 1 tới 5.
            # Đây là cách trực quan nhất để thấy văn bản đóng góp ở đâu, và nó
            # tái hiện đúng phát hiện của phần EDA: điểm số quyết định ở hai
            # đầu, còn ở vùng giữa thì nội dung mới là thứ phân định.
            with st.expander("Văn bản của bạn ảnh hưởng tới mức nào?", expanded=True):
                quet = []
                for d in [1.0, 2.0, 3.0, 4.0, 5.0]:
                    Xd = pd.DataFrame([{**{c: d for c in NUMERIC},
                                        "text_clean": sach}])
                    quet.append({"diem": d,
                                 "p": float(load_classifier().predict_proba(Xd)[0, 1])})
                qdf = pd.DataFrame(quet)

                duong = (
                    alt.Chart(qdf)
                    .mark_line(point=alt.OverlayMarkDef(size=70), color=SERIES[0])
                    .encode(
                        x=alt.X("diem:Q", title="Nếu mọi điểm số đều bằng",
                                scale=alt.Scale(domain=[0.8, 5.2]),
                                axis=alt.Axis(values=[1, 2, 3, 4, 5])),
                        y=alt.Y("p:Q", title="Xác suất Yes",
                                scale=alt.Scale(domain=[0, 1]),
                                axis=alt.Axis(format="%")),
                        tooltip=[alt.Tooltip("diem:Q", title="Điểm"),
                                 alt.Tooltip("p:Q", format=".1%", title="P(Yes)")],
                    )
                )
                moc = (
                    alt.Chart(pd.DataFrame({"y": [nguong]}))
                    .mark_rule(color=DANGER, strokeDash=[6, 4], size=2)
                    .encode(y="y:Q")
                )
                st.altair_chart((duong + moc).properties(height=230),
                                use_container_width=True)

                st.markdown(
                    f"""
Đường trên giữ nguyên **văn bản bạn vừa nhập** và chỉ thay đổi sáu điểm số.
Đường đứt đỏ là ngưỡng quyết định hiện tại ({nguong * 100:.0f}%).

Điểm số chi phối mạnh ở hai đầu: chấm 5 thì gần như luôn ra "Yes", chấm 1 thì gần
như luôn ra "No", bất kể viết gì. **Vùng quanh điểm 3 mới là nơi nội dung văn bản
quyết định.** Đúng như phần khảo sát dữ liệu đã chỉ ra: ở mức Rating 3 chỉ có 74,8%
người nói "Yes", nên điểm số một mình không phân biệt được và phải đọc nội dung.
                    """
                )

            st.info(
                "**Đây là dự đoán, không phải sự thật.** Mô hình bắt được khoảng "
                f"{m2['Recall_No'] * 100:.0f}% số trường hợp \"No\" trong tập kiểm tra, "
                f"và khi nó báo \"No\" thì đúng khoảng "
                f"{m2['Precision_No'] * 100:.0f}%. Vùng khó nhất là các đánh giá "
                "ở mức trung bình, nơi điểm số không phân biệt được và phải dựa "
                "vào nội dung văn bản."
            )

# ═══════════════════════════════════════════ 3. Hiệu năng mô hình
else:
    st.title("Hiệu năng mô hình")
    st.caption("Các con số dưới đây đo trên tập kiểm tra tách riêng, "
               "lấy trực tiếp từ notebook phân tích.")

    st.header("Yêu cầu 1 · Gợi ý công ty tương tự")
    st.markdown(
        """
Bài toán này **không có nhãn đúng**: dữ liệu không hề cho biết công ty nào giống
công ty nào. Cách đánh giá là dùng ba thuộc tính metadata làm **nhãn đối chứng**.
Ba thuộc tính này đã cố ý bị loại khỏi kho văn bản lúc huấn luyện, nên phép đo
không rơi vào vòng luẩn quẩn.
        """
    )

    attrs = ["Company industry", "Company Type", "Company size"]
    ten_vi = {"Company industry": "Lĩnh vực", "Company Type": "Loại hình",
              "Company size": "Quy mô"}
    hang = []
    for pp, r in meta["yeucau1"]["precision"].items():
        hang.append({"Phương pháp": pp,
                     **{ten_vi[a]: r[a] * 100 for a in attrs}})
    hang.append({"Phương pháp": "Chọn ngẫu nhiên",
                 **{ten_vi[a]: meta["yeucau1"]["baseline"][a] * 100
                    for a in attrs}})
    bang1 = pd.DataFrame(hang)

    dai = bang1.melt("Phương pháp", var_name="Thuộc tính", value_name="P@5")
    dai["Loại"] = np.where(dai["Phương pháp"] == "Chọn ngẫu nhiên",
                           "Mốc ngẫu nhiên", "Mô hình")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.altair_chart(
            alt.Chart(dai).mark_bar(cornerRadius=3).encode(
                x=alt.X("P@5:Q", title="Precision@5 (%)"),
                y=alt.Y("Phương pháp:N", title=None, sort=None),
                color=alt.Color("Loại:N",
                                scale=alt.Scale(domain=["Mô hình", "Mốc ngẫu nhiên"],
                                                range=[SERIES[0], DANGER]),
                                legend=alt.Legend(title=None, orient="top")),
                row=alt.Row("Thuộc tính:N", title=None,
                            header=alt.Header(labelFontWeight="bold")),
                tooltip=["Phương pháp", "Thuộc tính",
                         alt.Tooltip("P@5:Q", format=".1f")],
            ).properties(height=110),
            use_container_width=True,
        )
    with c2:
        st.dataframe(bang1.round(1), hide_index=True, use_container_width=True)
        st.caption(
            "Con số của **Loại hình** trông cao nhất nhưng gần như vô nghĩa, vì "
            "chỉ có ba loại hình và bốc ngẫu nhiên cũng đúng 43,9%. Ngược lại "
            "**Lĩnh vực** có tới 33 giá trị nên mốc ngẫu nhiên chỉ 12,8%, và đây "
            "mới là thuộc tính phân biệt được mô hình tốt với mô hình dở."
        )

    st.divider()
    st.header("Yêu cầu 2 · Dự đoán Recommend or Not")
    st.markdown(
        f"""
Nhãn mất cân bằng nặng, **{m2['yes_pct']}% Yes**. Hệ quả là **accuracy gây hiểu
nhầm**: chỉ cần đoán bừa "Yes" cho tất cả đã đạt
**{m2['baseline_accuracy'] * 100:.1f}%** mà không học được gì. Vì vậy chỉ số xếp
hạng chính là **F1 của lớp "No"**, lớp thiểu số và cũng là lớp nghiệp vụ quan tâm.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("F1 lớp \"No\"", f"{m2['F1_No']:.3f}",
              help="Chỉ số xếp hạng chính. Đoán bừa cho 0,000.")
    c2.metric("Recall lớp \"No\"", f"{m2['Recall_No']:.3f}")
    c3.metric("Precision lớp \"No\"", f"{m2['Precision_No']:.3f}")
    c4.metric("Accuracy", f"{m2['Accuracy']:.3f}",
              delta=f"{(m2['Accuracy'] - m2['baseline_accuracy']) * 100:+.1f} điểm "
                    f"so với đoán bừa")

    c1, c2 = st.columns([2, 3])
    with c1:
        st.subheader("Ma trận nhầm lẫn")
        cm = pd.DataFrame(
            [[m2["TN"], m2["FP"]], [m2["FN"], m2["TP"]]],
            index=["Thực tế No", "Thực tế Yes"],
            columns=["Dự đoán No", "Dự đoán Yes"],
        )
        st.dataframe(cm, use_container_width=True)
        st.caption(
            f"Trong {m2['TN'] + m2['FP']} review \"No\" của tập kiểm tra, mô hình "
            f"bắt đúng {m2['TN']} và bỏ sót {m2['FP']}."
        )
    with c2:
        st.subheader("Mô hình và cách chọn")
        st.markdown(
            f"""
| Hạng mục | Nội dung |
|---|---|
| Mô hình | {m2['model']} |
| Đặc trưng | TF-IDF 1 tới 2 từ (tối đa 20.000) kết hợp 6 cột điểm số |
| Dữ liệu | {m2['n_train']:,} review huấn luyện, {m2['n_test']:,} review kiểm tra |
| ROC-AUC | {m2['ROC_AUC']:.3f} |

Mô hình này đứng đầu trong **50 kết quả** đã thử, gồm 9 thuật toán scikit-learn và
6 thuật toán PySpark. Đáng chú ý là **không cách cân bằng lớp nào cải thiện được
F1**: cả `class_weight`, SMOTE lẫn hạ ngưỡng đều chỉ đánh đổi Precision lấy Recall.
            """
        )

    st.divider()
    st.subheader("Hạn chế cần nói rõ")
    st.markdown(
        """
- **Yêu cầu 1 không có nhãn thật.** Metadata chỉ là thước đo gián tiếp. Kiểm tra
  thủ công cho thấy Precision@5 đang **đánh giá thấp** chất lượng thật, vì nhiều
  gợi ý hợp lý bị tính là sai do lĩnh vực ghi khác tên.
- **Mô tả công ty là văn bản quảng bá** do chính công ty viết để thu hút ứng viên,
  nên có thể không phản ánh đúng thực tế công việc.
- **Dữ liệu trộn hai ngôn ngữ.** Review 88,9% tiếng Việt nhưng mô tả công ty chỉ
  31,2% tiếng Việt. Hai công ty cùng lĩnh vực mà viết bằng hai thứ tiếng khác nhau
  thì TF-IDF không khớp được.
- **Tập kiểm tra chỉ có 1.684 review**, trong đó khoảng 206 là "No". Chênh lệch
  dưới 0,02 F1 giữa các mô hình không nên coi là có ý nghĩa.
- **Chỉ 180 trong 478 công ty có đánh giá**, và FPT Software một mình chiếm 23,9%
  số review. Kết quả phản ánh chủ yếu các doanh nghiệp lớn.
        """
    )

    st.divider()
    st.caption(
        "**Lưu ý pháp lý.** Dữ liệu chỉ dùng cho mục đích học tập, không chia sẻ "
        "hoặc sử dụng cho mục đích khác."
    )
