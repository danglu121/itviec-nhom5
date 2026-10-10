"""ITViec: tìm công ty phù hợp và đọc hiểu đánh giá của nhân viên.

Ứng dụng dành cho người dùng cuối. Ứng viên tìm nơi làm việc hợp với mình,
nhà tuyển dụng soi các công ty cùng nhóm và đọc phản hồi của nhân viên.

Chạy:  streamlit run app.py
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
FIG = APP_DIR / "figures"

XANH, CAM, LUC = "#2a78d6", "#eb6834", "#1baf7a"
XANH_NHAT, XANH_DAM = "#9ec5f4", "#184f95"
DO, XAM = "#8f2222", "#52514e"

NUMERIC = ["Rating", "Salary & benefits", "Training & learning",
           "Management cares about me", "Culture & fun", "Office & workspace"]
NUMERIC_VI = {
    "Rating": "Mức hài lòng chung",
    "Salary & benefits": "Lương và phúc lợi",
    "Training & learning": "Cơ hội học hỏi",
    "Management cares about me": "Sếp quan tâm nhân viên",
    "Culture & fun": "Văn hoá và không khí",
    "Office & workspace": "Văn phòng và chỗ ngồi",
}

st.set_page_config(page_title="ITViec · Tìm công ty phù hợp",
                   page_icon="💼", layout="wide")


# ═══════════════════════════════════════════════════ Nạp dữ liệu
@st.cache_data
def load_companies() -> pd.DataFrame:
    return pd.read_parquet(ART / "companies.parquet")


@st.cache_data
def load_neighbors():
    z = np.load(ART / "neighbors_gensim.npz")
    return z["idx"].astype(int), z["score"]


@st.cache_data
def load_stats() -> pd.DataFrame:
    return pd.read_parquet(ART / "company_stats.parquet")


@st.cache_data
def load_monthly() -> pd.DataFrame:
    return pd.read_parquet(ART / "reviews_monthly.parquet")


@st.cache_data
def load_map() -> np.ndarray:
    return np.load(ART / "map_2d.npy")


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

# ════════════════════════════════════════════════════ Thanh bên
st.sidebar.image(str(FIG / "itviec_logo.png"), width=140)
st.sidebar.caption(f"Tìm công ty phù hợp trong {meta['n_companies']} công ty "
                   "công nghệ tại Việt Nam")
st.sidebar.divider()

vai = st.sidebar.radio("Bạn là ai?",
                       ["Tôi đang tìm việc", "Tôi là nhà tuyển dụng"])
LA_UNG_VIEN = vai == "Tôi đang tìm việc"

trang_1 = "Tìm công ty phù hợp" if LA_UNG_VIEN else "Công ty cùng nhóm với chúng tôi"
trang_2 = "Đọc hiểu một đánh giá" if LA_UNG_VIEN else "Phân tích phản hồi nhân viên"
page = st.sidebar.radio(
    "Chọn mục",
    [trang_1, trang_2, "Toàn cảnh thị trường IT", "Về ứng dụng"],
)

st.sidebar.divider()
st.sidebar.caption(
    f"Dựa trên mô tả của **{meta['n_companies']} công ty** và "
    f"**{m2['n_train'] + m2['n_test']:,} đánh giá thật** của nhân viên."
)

st.image(str(FIG / "hero_itviec.jpg"), use_container_width=True)
st.caption(
    "Đồ án sinh viên, **không liên kết với ITViec** và không được ITViec bảo "
    "trợ. Logo cùng hình ảnh thuộc về ITViec, sử dụng trong phạm vi học tập."
)


# ═══════════════════════════════════════════════ Thành phần dùng lại
def the_cong_ty(rows: pd.DataFrame, diem: np.ndarray, goc: pd.Series | None):
    """Danh sách công ty dạng thẻ, mỗi thẻ là một gợi ý."""
    for hang, ((_, r), d) in enumerate(zip(rows.iterrows(), diem), 1):
        cung_nganh = (goc is not None
                      and r["Company industry"] == goc["Company industry"])
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{hang}. {r['Company Name']}**")
            c2.markdown(
                f"<div style='text-align:right;color:{XAM};font-size:13px'>"
                f"độ hợp {d * 100:.0f}%</div>", unsafe_allow_html=True)
            st.caption(
                ("🟢 cùng lĩnh vực  ·  " if cung_nganh else "")
                + f"{r['Company industry']}  ·  {r['Company Type']}  ·  "
                + f"{r['Company size']}  ·  {r['Country']}"
            )
            # Gắn điểm đánh giá nếu công ty này từng có người chấm. Đây là
            # thứ người tìm việc muốn biết ngay, không nên bắt họ mở trang khác.
            dg = diem_danh_gia(r["id"])
            if dg is not None:
                m1, m2_, m3 = st.columns(3)
                m1.markdown(f"⭐ **{dg['Overall rating']:.1f}** / 5"
                            if pd.notna(dg["Overall rating"]) else "⭐ chưa có điểm")
                m2_.markdown(f"👍 **{dg['ty_le_gioi_thieu']:.0f}%** giới thiệu")
                m3.markdown(f"💬 **{int(dg['so_review'])}** đánh giá")
            else:
                st.markdown(
                    f"<span style='color:{XAM};font-size:13px'>"
                    "Chưa có nhân viên nào đánh giá công ty này</span>",
                    unsafe_allow_html=True)
            mo_ta = str(r["text_raw"])
            st.write(mo_ta[:340] + ("..." if len(mo_ta) > 340 else ""))
            # Cột trái phải đủ rộng, nếu không thì "Xem trên ITViec" bị
            # ngắt xuống hai dòng.
            a, b = st.columns([2, 3])
            if isinstance(r["Href"], str) and r["Href"].startswith("http"):
                a.markdown(f"[Xem trên ITViec]({r['Href']})")
            b.caption(f"Làm việc {r['Working days']}  ·  {r['Overtime Policy']}")


def diem_danh_gia(cty_id):
    """Trả về dòng thống kê của một công ty, hoặc None nếu chưa ai đánh giá."""
    st_df = load_stats()
    hit = st_df[st_df["id"] == cty_id]
    return None if hit.empty else hit.iloc[0]


def bieu_do_do_hop(rows: pd.DataFrame, diem: np.ndarray):
    d = pd.DataFrame({"ten": rows["Company Name"].to_numpy(), "diem": diem,
                      "nganh": rows["Company industry"].to_numpy()})
    d["nhan"] = [f"{i}. {t[:26]}" for i, t in enumerate(d["ten"], 1)]
    return (
        alt.Chart(d).mark_bar(cornerRadius=3, color=XANH, size=17).encode(
            x=alt.X("diem:Q", title="Độ hợp", axis=alt.Axis(format="%")),
            # labelOverlap=False buộc giữ đủ nhãn; để mặc định thì Altair lặng
            # lẽ bỏ bớt nhãn xen kẽ khi chiều cao không đủ.
            y=alt.Y("nhan:N", sort=None, title=None,
                    axis=alt.Axis(labelFontSize=11, labelLimit=230,
                                  labelOverlap=False)),
            tooltip=[alt.Tooltip("ten:N", title="Công ty"),
                     alt.Tooltip("nganh:N", title="Lĩnh vực"),
                     alt.Tooltip("diem:Q", format=".1%", title="Độ hợp")],
        ).properties(height=max(170, 38 * len(d)))
    )


def ban_do(i_goc: int | None, i_top: np.ndarray):
    """Bản đồ đặt toàn bộ công ty lên mặt phẳng theo nội dung mô tả."""
    xy = load_map()
    d = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1],
                      "ten": co["Company Name"], "nganh": co["Company industry"]})
    d["vai_tro"] = "Công ty khác"
    d.loc[i_top, "vai_tro"] = "Phù hợp với bạn"
    if i_goc is not None:
        d.loc[i_goc, "vai_tro"] = "Công ty bạn chọn"

    thu_tu = ["Công ty khác", "Phù hợp với bạn", "Công ty bạn chọn"]
    mau = ["#d8d7d3", CAM, XANH]
    # Vẽ ba lớp riêng để chấm nổi bật luôn nằm trên nền xám.
    lop = []
    for ten, kt in [("Công ty khác", 34), ("Phù hợp với bạn", 150),
                    ("Công ty bạn chọn", 320)]:
        sub = d[d["vai_tro"] == ten]
        if sub.empty:
            continue
        lop.append(
            alt.Chart(sub).mark_circle(
                size=kt, opacity=0.45 if ten == "Công ty khác" else 0.95).encode(
                x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
                color=alt.Color("vai_tro:N",
                                scale=alt.Scale(domain=thu_tu, range=mau),
                                legend=alt.Legend(title=None, orient="top")),
                tooltip=[alt.Tooltip("ten:N", title="Công ty"),
                         alt.Tooltip("nganh:N", title="Lĩnh vực")]))
    return alt.layer(*lop).properties(height=330).configure_view(strokeWidth=0)


# ═══════════════════════════════════════════ 1. Gợi ý công ty
if page in ("Tìm công ty phù hợp", "Công ty cùng nhóm với chúng tôi"):
    c1, c2 = st.columns([3, 1])
    with c1:
        if LA_UNG_VIEN:
            st.title("Tìm công ty phù hợp với bạn")
            st.write(
                "Bạn đã thích một công ty nào đó, hoặc đã hình dung ra nơi làm "
                "việc mong muốn? Ứng dụng sẽ tìm những công ty có mô tả gần "
                f"giống nhất trong {meta['n_companies']} công ty công nghệ đang "
                "tuyển tại Việt Nam."
            )
        else:
            st.title("Những công ty cùng nhóm với bạn")
            st.write(
                "Xem các công ty có định vị gần giống công ty bạn. Đây là nhóm "
                "bạn đang cạnh tranh để thu hút ứng viên, và cũng là nhóm ứng "
                "viên sẽ đem ra so sánh khi cân nhắc lời mời của bạn."
            )
    with c2:
        st.image(str(FIG / ("vai_ung_vien_anh.jpg" if LA_UNG_VIEN
                            else "vai_nha_tuyen_dung_anh.jpg")),
                 use_container_width=True)

    nhan_tab1 = "Bắt đầu từ một công ty" if LA_UNG_VIEN else "Chọn công ty của bạn"
    t1, t2 = st.tabs([nhan_tab1, "Mô tả nơi bạn muốn làm"])

    with t1:
        ten = st.selectbox(
            "Công ty", co["Company Name"].tolist(),
            index=int(co.index[co["Company Name"].str.contains(
                "FPT Software", na=False)][:1][0])
            if co["Company Name"].str.contains("FPT Software", na=False).any() else 0)
        k = st.slider("Số công ty muốn xem", 3, meta["topk"], 5, key="k1")

        i = int(co.index[co["Company Name"] == ten][0])
        goc = co.loc[i]

        with st.container(border=True):
            st.markdown(f"### {goc['Company Name']}")
            st.caption(f"{goc['Company industry']}  ·  {goc['Company Type']}  ·  "
                       f"{goc['Company size']}  ·  {goc['Country']}  ·  "
                       f"Làm việc {goc['Working days']}  ·  "
                       f"{goc['Overtime Policy']}")
            with st.expander("Công ty này giới thiệu về họ thế nào"):
                st.write(goc["text_raw"])

        st.subheader(f"{k} công ty gần giống nhất")
        nn_idx, nn_score = load_neighbors()
        top = nn_idx[i][:k]

        c1, c2 = st.columns([2, 3])
        with c1:
            st.altair_chart(bieu_do_do_hop(co.loc[top], nn_score[i][:k]),
                            use_container_width=True)
        with c2:
            st.altair_chart(ban_do(i, top), use_container_width=True)
            st.caption(
                "Mỗi chấm là một công ty. Hai công ty càng gần nhau trên bản đồ "
                "thì phần giới thiệu của họ càng giống nhau. Đây là hình minh "
                "hoạ để dễ hình dung, không phải thước đo chính xác."
            )

        st.caption("Chấm xanh lá trước tên nghĩa là cùng lĩnh vực với công ty "
                   "bạn chọn. Lĩnh vực không được đưa vào hệ thống, nên đây là "
                   "cách bạn tự kiểm chứng chất lượng gợi ý.")
        the_cong_ty(co.loc[top], nn_score[i][:k], goc)

    with t2:
        st.write(
            "Tả bằng lời nơi bạn muốn làm việc, tiếng Việt hay tiếng Anh đều "
            "được. Ứng dụng tự nhận ra ngôn ngữ bạn dùng."
            if LA_UNG_VIEN else
            "Tả định vị hoặc lĩnh vực của công ty bạn để tìm nhóm cạnh tranh."
        )
        goi_y_mau = {
            "Fintech và thanh toán":
                "Công ty fintech phát triển sản phẩm thanh toán, môi trường trẻ, "
                "có cơ hội học hỏi về blockchain và dữ liệu lớn",
            "Gia công phần mềm cho Nhật":
                "Công ty outsourcing làm dự án cho khách hàng Nhật Bản, có đào "
                "tạo tiếng Nhật, quy trình bài bản",
            "Sản phẩm AI và dữ liệu":
                "Product company làm về trí tuệ nhân tạo và xử lý dữ liệu lớn, "
                "đội ngũ kỹ sư mạnh, văn hoá cởi mở",
        }
        chon = st.radio("Chọn một mẫu có sẵn hoặc tự viết",
                        list(goi_y_mau) + ["Tự viết"], horizontal=True)
        mac_dinh = goi_y_mau.get(chon, "")
        truy_van = st.text_area("Mô tả của bạn", mac_dinh, height=110)
        k2 = st.slider("Số công ty muốn xem", 3, 20, 5, key="k2")

        if st.button("Tìm công ty", type="primary", use_container_width=True):
            sach, lang = preprocess(truy_van)
            if not sach.strip():
                st.warning("Chưa có nội dung để tìm. Hãy viết vài câu mô tả nơi "
                           "bạn muốn làm việc.")
            else:
                s = load_search()
                sim = cosine_similarity(s["vectorizer"].transform([sach]),
                                        s["matrix"]).ravel()
                top = np.argsort(-sim)[:k2]

                if sim[top[0]] < 0.05:
                    st.warning(
                        "Không tìm được công ty nào thật sự khớp. Thử viết dài "
                        "hơn, hoặc dùng những từ hay xuất hiện trong mô tả tuyển "
                        "dụng như tên công nghệ, lĩnh vực, quy mô đội ngũ."
                    )
                else:
                    st.caption(
                        f"Nhận ra bạn đang viết bằng "
                        f"**{'tiếng Việt' if lang == 'vi' else 'tiếng Anh'}**."
                    )
                    c1, c2 = st.columns([2, 3])
                    with c1:
                        st.altair_chart(bieu_do_do_hop(co.loc[top], sim[top]),
                                        use_container_width=True)
                    with c2:
                        st.altair_chart(ban_do(None, top),
                                        use_container_width=True)
                        st.caption("Vị trí các công ty phù hợp trên bản đồ.")
                    the_cong_ty(co.loc[top], sim[top], None)

# ═══════════════════════════════════ 2. Đọc hiểu đánh giá
elif page in ("Đọc hiểu một đánh giá", "Phân tích phản hồi nhân viên"):
    c1, c2 = st.columns([3, 1])
    with c1:
        if LA_UNG_VIEN:
            st.title("Đánh giá này nên hiểu thế nào?")
            st.write(
                "Đọc review về một công ty mà không rõ người viết thực sự hài "
                "lòng hay đang phàn nàn? Dán nội dung vào đây cùng số sao họ "
                "chấm. Ứng dụng cho biết đánh giá đó nghiêng về **giới thiệu** "
                "hay **không giới thiệu** công ty cho bạn bè."
            )
        else:
            st.title("Nhân viên này có giới thiệu công ty không?")
            st.write(
                "Dán một phản hồi của nhân viên cùng điểm họ chấm. Ứng dụng cho "
                "biết phản hồi đó nghiêng về **sẽ giới thiệu** hay **sẽ không "
                "giới thiệu** công ty cho bạn bè, giúp bạn phát hiện sớm dấu "
                "hiệu không hài lòng."
            )
    with c2:
        st.image(str(FIG / ("vai_ung_vien_anh.jpg" if LA_UNG_VIEN
                            else "vai_nha_tuyen_dung_anh.jpg")),
                 use_container_width=True)

    trai, phai = st.columns([3, 2])
    with trai:
        st.markdown("**Nội dung đánh giá**")
        tieu_de = st.text_input("Tiêu đề",
                                "Môi trường tốt nhưng lương chưa cạnh tranh")
        thich = st.text_area(
            "Điểm tốt được nhắc tới",
            "Đồng nghiệp thân thiện, sếp lắng nghe, được tham gia dự án lớn và "
            "học hỏi nhiều công nghệ mới. Văn phòng đẹp, giờ giấc linh hoạt.",
            height=120)
        gop_y = st.text_area(
            "Điểm chưa tốt hoặc góp ý",
            "Lương chưa theo kịp thị trường, quy trình review lương không rõ "
            "ràng. Thỉnh thoảng phải tăng ca khi gần deadline.",
            height=110)

    with phai:
        st.markdown("**Số sao người viết chấm** (1 tới 5)")
        diem = {c: st.slider(NUMERIC_VI[c], 1.0, 5.0, 4.0, 0.5, key=f"sl_{c}")
                for c in NUMERIC}

    if st.button("Xem kết quả", type="primary", use_container_width=True):
        tho = f"{tieu_de} {thich} {gop_y}".strip()
        sach, lang = preprocess(tho)

        if not sach.strip():
            st.warning("Chưa có nội dung đánh giá để phân tích.")
        else:
            X = pd.DataFrame([{**diem, "text_clean": sach}])
            p_yes = float(load_classifier().predict_proba(X)[0, 1])
            la_yes = p_yes >= 0.5
            mau = LUC if la_yes else CAM
            icon = "👍" if la_yes else "👎"
            tieu = ("Nghiêng về SẼ giới thiệu công ty"
                    if la_yes else "Nghiêng về KHÔNG giới thiệu công ty")
            chac_chan = ("rất rõ ràng" if abs(p_yes - 0.5) > 0.35
                         else "khá rõ" if abs(p_yes - 0.5) > 0.2
                         else "không chắc chắn lắm")

            st.divider()
            st.markdown(
                f"<div style='padding:20px;border-radius:12px;background:{mau}18;"
                f"border-left:6px solid {mau}'>"
                f"<div style='font-size:25px;font-weight:700;color:{mau}'>"
                f"{icon} {tieu}</div>"
                f"<div style='color:{XAM};margin-top:8px'>Mức độ: "
                f"<b>{chac_chan}</b>. Khả năng người viết sẽ giới thiệu công ty "
                f"là <b>{p_yes * 100:.0f}%</b>.</div></div>",
                unsafe_allow_html=True)

            st.write("")
            c1, c2, c3 = st.columns(3)
            c1.metric("Khả năng sẽ giới thiệu", f"{p_yes * 100:.0f}%")
            c2.metric("Số sao trung bình",
                      f"{np.mean(list(diem.values())):.1f} / 5")
            c3.metric("Ngôn ngữ",
                      "Tiếng Việt" if lang == "vi" else "Tiếng Anh")

            thanh = pd.DataFrame({"nhan": ["Không giới thiệu", "Sẽ giới thiệu"],
                                  "p": [1 - p_yes, p_yes]})
            nen = alt.Chart(thanh).encode(
                y=alt.Y("nhan:N", sort=["Không giới thiệu", "Sẽ giới thiệu"],
                        title=None,
                        axis=alt.Axis(labelFontSize=13, labelPadding=10)))
            cot = nen.mark_bar(cornerRadius=4, size=34).encode(
                x=alt.X("p:Q", scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format="%"), title="Khả năng"),
                color=alt.Color("nhan:N", scale=alt.Scale(
                    domain=["Không giới thiệu", "Sẽ giới thiệu"],
                    range=[CAM, LUC]), legend=None))
            # Nhãn số đặt cạnh đầu cột. Với khả năng rất nhỏ thì cột gần như vô
            # hình, lúc đó chỉ còn con số cho biết giá trị thật.
            chu = nen.mark_text(align="left", dx=7, fontSize=12,
                                color=XAM).encode(
                x=alt.X("p:Q", scale=alt.Scale(domain=[0, 1])),
                text=alt.Text("p:Q", format=".0%"))
            st.altair_chart((cot + chu).properties(height=150),
                            use_container_width=True)

            with st.expander("Số sao hay nội dung quyết định kết quả?",
                             expanded=True):
                quet = []
                for d in [1.0, 2.0, 3.0, 4.0, 5.0]:
                    Xd = pd.DataFrame([{**{c: d for c in NUMERIC},
                                        "text_clean": sach}])
                    quet.append({"diem": d, "p": float(
                        load_classifier().predict_proba(Xd)[0, 1])})
                qdf = pd.DataFrame(quet)

                duong = alt.Chart(qdf).mark_line(
                    point=alt.OverlayMarkDef(size=70), color=XANH).encode(
                    x=alt.X("diem:Q", title="Nếu người viết chấm tất cả bằng",
                            scale=alt.Scale(domain=[0.8, 5.2]),
                            axis=alt.Axis(values=[1, 2, 3, 4, 5])),
                    y=alt.Y("p:Q", title="Khả năng sẽ giới thiệu",
                            scale=alt.Scale(domain=[0, 1]),
                            axis=alt.Axis(format="%")),
                    tooltip=[alt.Tooltip("diem:Q", title="Số sao"),
                             alt.Tooltip("p:Q", format=".0%", title="Khả năng")])
                moc = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
                    color=DO, strokeDash=[6, 4], size=2).encode(y="y:Q")
                st.altair_chart((duong + moc).properties(height=230),
                                use_container_width=True)
                st.write(
                    "Đường trên giữ nguyên nội dung bạn vừa dán và chỉ thay đổi "
                    "số sao. Khi chấm 5 sao thì gần như luôn là sẽ giới thiệu, "
                    "chấm 1 sao thì gần như luôn ngược lại, bất kể viết gì. "
                    "**Vùng quanh 3 sao mới là nơi nội dung quyết định**, và đó "
                    "cũng là lúc công cụ này hữu ích nhất."
                )

            st.info(
                "Đây là dự đoán, không phải sự thật. Trong các đánh giá thực tế "
                f"thuộc nhóm không giới thiệu, ứng dụng nhận ra được khoảng "
                f"{m2['Recall_No'] * 100:.0f}%. Khi ứng dụng báo không giới "
                f"thiệu thì đúng khoảng {m2['Precision_No'] * 100:.0f}%."
            )

# ═══════════════════════════════════ 3. Toàn cảnh thị trường
elif page == "Toàn cảnh thị trường IT":
    st.title("Toàn cảnh thị trường IT Việt Nam")
    st.write(
        f"Vài con số về {meta['n_companies']} công ty công nghệ đang tuyển "
        "dụng, giúp bạn hình dung thị trường trước khi chọn nơi ứng tuyển."
        if LA_UNG_VIEN else
        f"Vài con số về {meta['n_companies']} công ty công nghệ, giúp bạn định "
        "vị mình giữa thị trường."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Công ty", f"{len(co)}")
    c2.metric("Lĩnh vực", f"{co['Company industry'].nunique()}")
    c3.metric("Công ty Việt Nam", f"{(co['Country'] == 'Vietnam').sum()}")
    c4.metric("Khai báo No OT",
              f"{(co['Overtime Policy'] == 'No OT').sum()}")

    st.caption(
        "Lưu ý: chính sách tăng ca là do công ty tự khai khi đăng tin, nên nên "
        "xem như lời giới thiệu chứ không phải thực tế đã được kiểm chứng."
    )

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Lĩnh vực nào nhiều công ty nhất")
        vc = (co["Company industry"].value_counts().head(12)
              .rename_axis("linh_vuc").reset_index(name="so_cty"))
        st.altair_chart(
            alt.Chart(vc).mark_bar(cornerRadius=3, color=XANH).encode(
                x=alt.X("so_cty:Q", title="Số công ty"),
                y=alt.Y("linh_vuc:N", sort="-x", title=None,
                        axis=alt.Axis(labelLimit=250, labelOverlap=False)),
                tooltip=[alt.Tooltip("linh_vuc:N", title="Lĩnh vực"),
                         alt.Tooltip("so_cty:Q", title="Số công ty")],
            ).properties(height=330), use_container_width=True)

    with c2:
        st.subheader("Quy mô đội ngũ")
        thu_tu_qm = ["1-50 employees", "51-150 employees", "151-300 employees",
                     "301-500 employees", "501-1000 employees", "1000+ employees"]
        qm = (co["Company size"].value_counts()
              .rename_axis("quy_mo").reset_index(name="so_cty"))
        qm["quy_mo"] = pd.Categorical(qm["quy_mo"], thu_tu_qm, ordered=True)
        qm = qm.dropna(subset=["quy_mo"]).sort_values("quy_mo")
        st.altair_chart(
            alt.Chart(qm).mark_bar(cornerRadius=3, color=XANH_DAM).encode(
                x=alt.X("so_cty:Q", title="Số công ty"),
                y=alt.Y("quy_mo:N", sort=thu_tu_qm, title=None,
                        axis=alt.Axis(labelLimit=200, labelOverlap=False)),
                tooltip=[alt.Tooltip("quy_mo:N", title="Quy mô"),
                         alt.Tooltip("so_cty:Q", title="Số công ty")],
            ).properties(height=330), use_container_width=True)
        st.caption("Phần lớn là công ty nhỏ và vừa. Nếu bạn muốn môi trường "
                   "đông người thì nhóm trên 1000 nhân sự chỉ có vài chục lựa chọn.")

    st.divider()
    st.subheader("Tra điểm đánh giá của một công ty")
    st_df = load_stats()
    KHIA = ["Salary & benefits", "Training & learning",
            "Management cares about me", "Culture & fun", "Office & workspace"]
    KHIA_VI = {"Salary & benefits": "Lương và phúc lợi",
               "Training & learning": "Cơ hội học hỏi",
               "Management cares about me": "Sếp quan tâm nhân viên",
               "Culture & fun": "Văn hoá và không khí",
               "Office & workspace": "Văn phòng và chỗ ngồi"}

    ten_tra = st.selectbox(
        f"Chọn công ty, {len(st_df)} công ty đã có người đánh giá",
        st_df.sort_values("so_review", ascending=False)["Company Name"].tolist())
    r = st_df[st_df["Company Name"] == ten_tra].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Điểm chung",
              f"{r['Overall rating']:.1f} / 5" if pd.notna(r["Overall rating"])
              else "chưa có")
    c2.metric("Tỷ lệ sẽ giới thiệu", f"{r['ty_le_gioi_thieu']:.0f}%")
    c3.metric("Số người đánh giá", f"{int(r['so_review'])}")
    c4.metric("Đánh giá gần nhất", f"{pd.to_datetime(r['review_cuoi']):%m/%Y}")

    if r["so_review"] < 10:
        st.warning(
            f"Chỉ {int(r['so_review'])} người đánh giá công ty này. Với số "
            "lượng ít như vậy, một ý kiến cực đoan cũng đủ kéo lệch điểm trung "
            "bình. Hãy đọc kỹ từng đánh giá thay vì nhìn con số tổng."
        )

    # So điểm từng khía cạnh với mức giữa của toàn thị trường, để người xem
    # biết công ty này mạnh yếu ở đâu chứ không chỉ thấy một con số trần trụi.
    trung_vi = {k: st_df[k].median() for k in KHIA}
    ss = pd.DataFrame([
        {"khia_canh": KHIA_VI[k], "nhom": "Công ty này", "diem": r[k]}
        for k in KHIA if pd.notna(r[k])
    ] + [
        {"khia_canh": KHIA_VI[k], "nhom": "Mức giữa thị trường",
         "diem": trung_vi[k]} for k in KHIA if pd.notna(r[k])
    ])
    if ss.empty:
        st.info("Công ty này chưa có điểm cho từng khía cạnh.")
    else:
        st.altair_chart(
            alt.Chart(ss).mark_bar(cornerRadius=3).encode(
                # tickCount thưa lại, nếu không Altair chia tới 26 vạch
                x=alt.X("diem:Q", title="Điểm (thang 1 tới 5)",
                        scale=alt.Scale(domain=[0, 5]),
                        axis=alt.Axis(tickCount=6)),
                y=alt.Y("khia_canh:N", sort=None, title=None,
                        axis=alt.Axis(labelLimit=220, labelOverlap=False)),
                yOffset="nhom:N",
                color=alt.Color("nhom:N", scale=alt.Scale(
                    domain=["Công ty này", "Mức giữa thị trường"],
                    range=[XANH, "#c9ccd1"]),
                    legend=alt.Legend(title=None, orient="top")),
                tooltip=[alt.Tooltip("khia_canh:N", title="Khía cạnh"),
                         alt.Tooltip("nhom:N", title=""),
                         alt.Tooltip("diem:Q", format=".2f", title="Điểm")],
            ).properties(height=290), use_container_width=True)

    st.divider()
    st.subheader("Điểm cao có đi kèm nhiều người đánh giá không?")
    nganh_chon = st.multiselect(
        "Lọc theo lĩnh vực, để trống là xem tất cả",
        sorted(st_df["Company industry"].dropna().unique()))
    loc = st_df if not nganh_chon else st_df[st_df["Company industry"].isin(nganh_chon)]
    loc = loc.dropna(subset=["Overall rating"])

    st.altair_chart(
        alt.Chart(loc).mark_circle(opacity=0.72).encode(
            x=alt.X("so_review:Q", title="Số người đánh giá",
                    scale=alt.Scale(type="log")),
            y=alt.Y("Overall rating:Q", title="Điểm chung",
                    scale=alt.Scale(domain=[2, 5])),
            size=alt.Size("ty_le_gioi_thieu:Q", title="% giới thiệu",
                          scale=alt.Scale(range=[25, 320])),
            color=alt.Color("Company Type:N", title=None,
                            scale=alt.Scale(scheme="tableau10"),
                            legend=alt.Legend(orient="top", columns=2,
                                              labelLimit=260)),
            tooltip=[alt.Tooltip("Company Name:N", title="Công ty"),
                     alt.Tooltip("Company industry:N", title="Lĩnh vực"),
                     alt.Tooltip("so_review:Q", title="Số đánh giá"),
                     alt.Tooltip("Overall rating:Q", format=".2f", title="Điểm"),
                     alt.Tooltip("ty_le_gioi_thieu:Q", format=".0f",
                                 title="% giới thiệu")],
        ).properties(height=400).interactive(), use_container_width=True)
    st.caption(
        f"Đang hiện {len(loc)} công ty. Trục ngang dùng thang log vì số đánh "
        "giá chênh nhau rất lớn, từ 3 tới hơn hai nghìn. Rê chuột lên từng "
        "chấm để xem tên, cuộn để phóng to, kéo để di chuyển. "
        "**Các chấm bên trái là công ty có rất ít đánh giá**, điểm của họ kém "
        "tin cậy hơn dù cao hay thấp."
    )

    st.divider()
    st.subheader("Dữ liệu này mới tới đâu")
    th = load_monthly()
    th["nam"] = pd.to_datetime(th["Cmt_date"]).dt.year
    nam_min, nam_max = int(th["nam"].min()), int(th["nam"].max())
    khoang = st.slider("Khoảng thời gian", nam_min, nam_max,
                       (nam_min, nam_max))
    tv = th[(th["nam"] >= khoang[0]) & (th["nam"] <= khoang[1])]

    c1, c2 = st.columns([3, 1])
    with c1:
        st.altair_chart(
            alt.Chart(tv).mark_area(
                line={"color": XANH}, opacity=0.25, color=XANH).encode(
                x=alt.X("Cmt_date:T", title="Thời gian"),
                y=alt.Y("so_review:Q", title="Số đánh giá mỗi tháng"),
                tooltip=[alt.Tooltip("Cmt_date:T", title="Tháng",
                                     format="%m/%Y"),
                         alt.Tooltip("so_review:Q", title="Số đánh giá"),
                         alt.Tooltip("ty_le_gioi_thieu:Q", format=".0f",
                                     title="% giới thiệu")],
            ).properties(height=280).interactive(), use_container_width=True)
    with c2:
        st.metric("Đánh giá trong khoảng", f"{int(tv['so_review'].sum()):,}")
        st.metric("Tỷ lệ giới thiệu trung bình",
                  f"{tv['ty_le_gioi_thieu'].mean():.0f}%")
        st.caption("Một công ty có thể đã thay đổi nhiều so với lúc người ta "
                   "viết đánh giá, nhất là các năm đầu.")

    st.divider()
    st.subheader("Người ta viết đánh giá bằng ngôn ngữ nào")
    # Cột trái rộng hơn cột phải: nhãn "Nhân viên đánh giá" và tiêu đề trục
    # đều dài, cột hẹp sẽ cắt mất chữ.
    c1, c2 = st.columns([3, 2])
    with c1:
        ngon_ngu = pd.DataFrame({
            "nguon": ["Công ty tự giới thiệu", "Nhân viên đánh giá"],
            "vi": [31.2, 88.9]})
        nen_nn = alt.Chart(ngon_ngu).encode(
            y=alt.Y("nguon:N", sort=None, title=None,
                    axis=alt.Axis(labelLimit=220, labelOverlap=False)))
        cot_nn = nen_nn.mark_bar(cornerRadius=3, color=XANH, size=30).encode(
            x=alt.X("vi:Q", title="Viết bằng tiếng Việt (%)",
                    scale=alt.Scale(domain=[0, 100])),
            tooltip=[alt.Tooltip("nguon:N", title="Nguồn"),
                     alt.Tooltip("vi:Q", format=".1f", title="% tiếng Việt")])
        chu_nn = nen_nn.mark_text(align="left", dx=7, fontSize=12,
                                  color=XAM).encode(
            x="vi:Q", text=alt.Text("vi:Q", format=".1f"))
        st.altair_chart((cot_nn + chu_nn).properties(height=160),
                        use_container_width=True)
    with c2:
        st.markdown(
            """
Một điểm thú vị của thị trường này: **công ty giới thiệu mình chủ yếu bằng
tiếng Anh, còn nhân viên đánh giá chủ yếu bằng tiếng Việt.**

Điều đó có nghĩa là khi bạn tìm bằng tiếng Việt ở mục **Mô tả nơi bạn muốn
làm**, kết quả có thể bỏ sót những công ty phù hợp nhưng viết giới thiệu bằng
tiếng Anh. Thử tìm lại bằng tiếng Anh nếu kết quả chưa như ý.
            """
        )

# ═══════════════════════════════════════════════ 4. Về ứng dụng
else:
    st.title("Về ứng dụng")

    st.image(str(FIG / "itviec_minhhoa.jpg"), use_container_width=True)

    st.header("Ứng dụng này làm được gì")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
**Nếu bạn đang tìm việc**

Tìm công ty gần giống một nơi bạn đã thích, hoặc mô tả bằng lời nơi bạn muốn
làm để hệ thống tìm giúp. Bạn cũng có thể dán một review tìm được trên mạng để
xem nó nghiêng về khen hay chê.
            """
        )
    with c2:
        st.markdown(
            """
**Nếu bạn là nhà tuyển dụng**

Xem nhóm công ty có định vị gần giống bạn, tức nhóm bạn đang cạnh tranh để thu
hút ứng viên. Dán phản hồi của nhân viên để phát hiện sớm dấu hiệu không hài
lòng.
            """
        )

    st.divider()
    st.header("Số liệu đến từ đâu")
    c1, c2, c3 = st.columns(3)
    c1.metric("Công ty", f"{meta['n_companies']}")
    c1.caption("có phần giới thiệu đủ dài để so sánh")
    c2.metric("Đánh giá của nhân viên",
              f"{m2['n_train'] + m2['n_test']:,}")
    c2.caption("thu thập từ 2016 tới 2025")
    c3.metric("Độ chính xác khi nhận ra người không hài lòng",
              f"{m2['Precision_No'] * 100:.0f}%")
    c3.caption("trong số lần báo không giới thiệu")

    st.divider()
    st.header("Khi nào không nên tin kết quả")
    st.markdown(
        """
- **Công ty tự viết phần giới thiệu của mình.** Đó là văn bản quảng bá để thu
  hút ứng viên, nên hai công ty gần nhau trong gợi ý nghĩa là họ *nói* giống
  nhau, chưa chắc *làm việc* giống nhau.
- **Chỉ 180 trong 478 công ty có người đánh giá**, và riêng một công ty lớn đã
  chiếm gần một phần tư số đánh giá. Bức tranh nghiêng về doanh nghiệp lớn.
- **Rào cản ngôn ngữ.** Mô tả công ty phần lớn bằng tiếng Anh còn đánh giá phần
  lớn bằng tiếng Việt. Hệ thống so khớp theo từ ngữ, nên hai công ty cùng lĩnh
  vực mà viết bằng hai thứ tiếng khác nhau có thể không nhận ra nhau.
- **Đánh giá là cảm nhận cá nhân** tại một thời điểm, của một người, ở một vị
  trí cụ thể. Công ty thay đổi theo thời gian.
        """
    )

    st.divider()
    with st.expander("Chi tiết kỹ thuật, dành cho người quan tâm"):
        st.markdown(
            f"""
**Gợi ý công ty.** Content-Based Filtering trên phần giới thiệu của
{meta['n_companies']} công ty. Năm phương pháp đã được so sánh, gồm TF-IDF với
cosine của scikit-learn, TF-IDF của Gensim, LSI, BM25 Okapi, và Doc2Vec. Phương
pháp thắng là **TF-IDF của Gensim**.

Bài toán này **không có nhãn đúng**: dữ liệu không hề cho biết công ty nào
giống công ty nào. Cách đánh giá là dùng lĩnh vực, loại hình và quy mô làm nhãn
đối chứng. Ba cột này đã cố ý bị loại khỏi phần văn bản đưa vào mô hình, nên
phép đo không rơi vào vòng luẩn quẩn.

**Dự đoán Recommend.** {m2['model']}, huấn luyện trên {m2['n_train']:,} đánh
giá và kiểm tra trên {m2['n_test']:,} đánh giá chưa từng thấy. Nhãn mất cân
bằng nặng, {m2['yes_pct']} phần trăm là sẽ giới thiệu, nên chỉ số xếp hạng
chính là F1 của lớp thiểu số chứ không phải accuracy: đoán bừa sẽ giới thiệu
cho tất cả đã đạt accuracy {m2['baseline_accuracy'] * 100:.1f} phần trăm mà
không học được gì.
            """
        )

        p_tot_nhat = max(v["Company industry"]
                         for v in meta["yeucau1"]["precision"].values())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P@5 lĩnh vực", f"{p_tot_nhat * 100:.1f}%")
        c1.caption(f"mốc ngẫu nhiên "
                   f"{meta['yeucau1']['baseline']['Company industry'] * 100:.1f}%")
        c2.metric("F1 lớp thiểu số", f"{m2['F1_No']:.3f}")
        c2.caption("đoán bừa cho 0,000")
        c3.metric("ROC-AUC", f"{m2['ROC_AUC']:.3f}")
        c4.metric("Accuracy", f"{m2['Accuracy']:.3f}")
        c4.caption(f"đoán bừa đạt {m2['baseline_accuracy'] * 100:.1f}%")

        f0, f1, f2, f3 = st.tabs(["Đánh giá phần gợi ý công ty",
                                  "So sánh 15 mô hình", "Xử lý mất cân bằng",
                                  "Văn bản đóng góp gì"])
        with f0:
            st.image(str(FIG / "09_yc1_precision.png"), use_container_width=True)
            st.write(
                "Cả năm phương pháp đều vượt mốc chọn ngẫu nhiên trên cả ba "
                "thuộc tính đối chứng. Con số của Loại hình trông cao nhất "
                "nhưng gần như vô nghĩa vì chỉ có ba loại hình, bốc bừa cũng "
                "đúng 43,9 phần trăm. Lĩnh vực có 33 giá trị nên mốc ngẫu "
                "nhiên chỉ 12,8 phần trăm, và đó mới là thuộc tính phân biệt "
                "được phương pháp tốt với phương pháp dở."
            )
            st.image(str(FIG / "10_yc1_overlap.png"), use_container_width=True)
            st.write(
                "Hai cài đặt TF-IDF của hai thư viện trùng nhau 0,81 số gợi ý, "
                "xác nhận cả hai được cấu hình đúng. BM25 tuy cũng đếm từ mà "
                "chỉ trùng khoảng 0,46. Doc2Vec gần như không đồng thuận với "
                "ai, chỉ 0,13 tới 0,16, và đó là dấu hiệu xấu vì nó cũng có "
                "điểm thấp nhất."
            )
        with f1:
            st.image(str(FIG / "12_yc2_comparison.png"), use_container_width=True)
            st.write(
                "Bên trái, F1 của lớp thiểu số trải rộng từ 0 tới 0,74 nên phân "
                "biệt rõ mô hình tốt và mô hình hỏng. Bên phải, accuracy chen "
                "chúc quanh 0,88 tới 0,94, nhiều mô hình chỉ nhỉnh hơn vạch đỏ "
                "đoán bừa vài phần trăm. Đây là lý do accuracy không được dùng "
                "làm chỉ số chính."
            )
            st.image(str(FIG / "15_yc2_confusion_roc.png"),
                     use_container_width=True)
        with f2:
            st.image(str(FIG / "13_yc2_imbalance.png"), use_container_width=True)
            st.write(
                "Cả class_weight lẫn SMOTE đều làm tăng Recall nhưng làm giảm "
                "F1, nghĩa là phần Precision mất đi lớn hơn phần Recall thu "
                "được. Không cách rebalancing nào cải thiện được F1 trên dữ "
                "liệu này."
            )
        with f3:
            st.image(str(FIG / "14_yc2_ablation.png"), use_container_width=True)
            st.write(
                "Số sao mạnh hơn văn bản khi đứng một mình, nhưng kết hợp cả "
                "hai thì tốt hơn cả hai. Phần thông tin mà văn bản đóng góp nằm "
                "ở vùng quanh 3 sao, đúng như biểu đồ độ nhạy ở trang đọc hiểu "
                "đánh giá cho thấy."
            )

    st.divider()
    st.markdown(
        """
**Đồ án tốt nghiệp Data Science, Nhóm 5**  
Lư Trần Nguyên Đăng, Nguyễn Anh Duy, Phạm Thị Phương Thúy  
Trung Tâm Tin Học, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM

Đây là **bài tập của sinh viên**, không liên kết với ITViec, không được ITViec
bảo trợ hay kiểm duyệt. Tên, logo và hình ảnh ITViec thuộc quyền sở hữu của
ITViec, được sử dụng ở đây trong phạm vi học tập để minh hoạ cho bộ dữ liệu mà
nhà trường cung cấp.

Dữ liệu chỉ dùng cho mục đích học tập, không chia sẻ hoặc sử dụng cho mục đích
khác.
        """
    )
