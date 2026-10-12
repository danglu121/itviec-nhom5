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

# Mỗi mục có một màu và một hình riêng. Người dùng nhìn dải đầu trang là biết
# mình đang ở đâu, không phải đọc lại tiêu đề hay nhìn thanh bên.
MAU_MUC = {
    "Tìm công ty phù hợp": {
        "mau": "#2a78d6", "icon": "icon_tim_cty.png",
        "tua": "Tìm công ty phù hợp với bạn",
        "dan": "Bắt đầu từ một công ty bạn đã thích, hoặc tả bằng lời nơi bạn "
               "muốn làm. Ứng dụng tìm những công ty có phần giới thiệu gần "
               "giống nhất.",
    },
    "Công ty cùng nhóm với chúng tôi": {
        "mau": "#2a78d6", "icon": "icon_tim_cty.png",
        "tua": "Những công ty cùng nhóm với bạn",
        "dan": "Xem các công ty có định vị gần giống công ty bạn. Đây là nhóm "
               "bạn đang cạnh tranh để thu hút ứng viên, và cũng là nhóm ứng "
               "viên đem ra so sánh khi cân nhắc lời mời của bạn.",
    },
    "Đọc hiểu một đánh giá": {
        "mau": "#eb6834", "icon": "icon_doc_danh_gia.png",
        "tua": "Đánh giá này nên hiểu thế nào?",
        "dan": "Dán nội dung một đánh giá cùng số sao người viết chấm. Ứng "
               "dụng cho biết đánh giá đó nghiêng về giới thiệu hay không "
               "giới thiệu công ty cho bạn bè.",
    },
    "Phân tích phản hồi nhân viên": {
        "mau": "#eb6834", "icon": "icon_doc_danh_gia.png",
        "tua": "Nhân viên này có giới thiệu công ty không?",
        "dan": "Dán một phản hồi của nhân viên cùng điểm họ chấm. Ứng dụng "
               "cho biết phản hồi đó nghiêng về sẽ giới thiệu hay không, giúp "
               "bạn phát hiện sớm dấu hiệu không hài lòng.",
    },
    "Toàn cảnh thị trường IT": {
        "mau": "#1b9e74", "icon": "icon_thi_truong.png",
        "tua": "Toàn cảnh thị trường IT Việt Nam",
        "dan": "Vài con số về các công ty công nghệ đang tuyển dụng, để bạn "
               "hình dung thị trường trước khi chọn nơi ứng tuyển hoặc định "
               "vị công ty mình giữa thị trường.",
    },
    "Về ứng dụng": {
        "mau": "#46617d", "icon": "icon_ve_ung_dung.png",
        "tua": "Về ứng dụng",
        "dan": "Số liệu đến từ đâu, ứng dụng làm được gì, chính xác tới mức "
               "nào và khi nào thì không nên tin kết quả.",
    },
}


def dai_dau_trang(khoa: str, anh_phu: str | None = None) -> str:
    """Dải tiêu đề có màu riêng và hình minh hoạ của mục. Trả về màu của mục.

    `anh_phu` là ảnh thật của vai đang chọn, đặt bên phải dải tiêu đề.
    """
    ty_le = [1, 4.3, 1.4] if anh_phu else [1, 5.4]
    cot = st.columns(ty_le, vertical_alignment="center")
    c1, c2 = cot[0], cot[1]
    if anh_phu:
        cot[2].image(str(FIG / anh_phu), use_container_width=True)
    m = MAU_MUC[khoa]
    c1.image(str(FIG / m["icon"]), use_container_width=True)
    c2.markdown(
        f"<div style='padding:16px 22px;border-radius:13px;"
        f"background:linear-gradient(95deg,{m['mau']}22,{m['mau']}06);"
        f"border-left:7px solid {m['mau']}'>"
        f"<div style='font-size:29px;font-weight:800;color:{m['mau']};"
        f"line-height:1.18'>{m['tua']}</div>"
        f"<div style='font-size:14.5px;color:{XAM};margin-top:7px;"
        f"line-height:1.5'>{m['dan']}</div></div>",
        unsafe_allow_html=True)
    return m["mau"]


def o_so_lieu(cot, nhan: str, gia_tri: str, phu: str = "", mau: str = XANH):
    """Một ô số liệu có màu. st.metric chỉ có chữ đen, đọc nhiều ô thì rối."""
    # Giá trị dài, chẳng hạn tên công ty, phải thu nhỏ lại thì mới nằm vừa ô.
    co_chu = 27 if len(gia_tri) <= 11 else (22 if len(gia_tri) <= 16 else 18)
    cot.markdown(
        f"<div style='padding:13px 15px;border-radius:12px;background:{mau}14;"
        f"border-top:4px solid {mau};height:100%'>"
        f"<div style='font-size:12.5px;color:{XAM};font-weight:600;"
        f"line-height:1.3'>{nhan}</div>"
        f"<div style='font-size:{co_chu}px;font-weight:800;color:{mau};"
        f"line-height:1.3;margin-top:2px'>{gia_tri}</div>"
        f"<div style='font-size:12px;color:{XAM};line-height:1.35'>{phu}</div>"
        f"</div>", unsafe_allow_html=True)


def hang_so_lieu(muc: list[tuple], mau: str, ty_le: list | None = None):
    """Một hàng ô số liệu cùng màu, mỗi phần tử là (nhãn, giá trị, chú thích)."""
    cols = st.columns(ty_le or len(muc))
    for col, m in zip(cols, muc):
        o_so_lieu(col, m[0], m[1], m[2] if len(m) > 2 else "", mau)


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

# Thông tin nhóm đặt cuối thanh bên, dưới phần điều hướng, để không chen
# vào luồng thao tác của người dùng.
st.sidebar.divider()
st.sidebar.markdown("**Thông tin liên hệ**")
for _ten, _mail in [
    ("Lư Trần Nguyên Đăng", "lutrannguyendang@gmail.com"),
    ("Phạm Thị Phương Thúy", "phuongthuy219@gmail.com"),
    ("Nguyễn Anh Duy", "toulouse18uef@gmail.com"),
]:
    st.sidebar.markdown(
        f"<div style='line-height:1.35;margin-bottom:9px'>"
        f"<span style='font-size:13px'>{_ten}</span><br>"
        f"<a href='mailto:{_mail}' style='font-size:11.5px;color:{XAM};"
        f"text-decoration:none'>{_mail}</a></div>",
        unsafe_allow_html=True)
st.sidebar.caption(
    "Trung Tâm Tin Học, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM")

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


def zoom_giu_shift():
    """Phóng to bằng con lăn, nhưng phải giữ Shift.

    Để .interactive() mặc định thì biểu đồ nuốt mọi thao tác con lăn: người
    dùng tưởng đang cuộn trang mà hoá ra đang phóng to biểu đồ, và trục bị
    kéo ra những khoảng vô nghĩa. Buộc giữ Shift thì cuộn trang vẫn bình
    thường, ai muốn phóng to thì vẫn phóng được.
    """
    return alt.selection_interval(bind="scales",
                                  zoom="wheel![event.shiftKey]")


def diem_danh_gia(cty_id):
    """Trả về dòng thống kê của một công ty, hoặc None nếu chưa ai đánh giá."""
    st_df = load_stats()
    hit = st_df[st_df["id"] == cty_id]
    return None if hit.empty else hit.iloc[0]


def bieu_do_do_hop(rows: pd.DataFrame, diem: np.ndarray, mau: str = XANH):
    d = pd.DataFrame({"ten": rows["Company Name"].to_numpy(), "diem": diem,
                      "nganh": rows["Company industry"].to_numpy()})
    d["nhan"] = [f"{i}. {t[:26]}" for i, t in enumerate(d["ten"], 1)]
    return (
        alt.Chart(d).mark_bar(cornerRadius=3, color=mau, size=17).encode(
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


THU_TU_QM = ["1-50 employees", "51-150 employees", "151-300 employees",
             "301-500 employees", "501-1000 employees", "1000+ employees"]

KHIA = ["Salary & benefits", "Training & learning",
        "Management cares about me", "Culture & fun", "Office & workspace"]
KHIA_VI = {k: NUMERIC_VI[k] for k in KHIA}


def bo_loc_cong_ty(khoa: str) -> dict:
    """Bộ lọc dùng chung cho hai cách tìm. Lọc xong thì kết quả vẽ lại ngay."""
    with st.expander("Lọc kết quả theo tiêu chí của bạn", expanded=False):
        c1, c2, c3 = st.columns(3)
        nganh = c1.multiselect(
            "Lĩnh vực", sorted(co["Company industry"].dropna().unique()),
            key=f"{khoa}_nganh")
        quy_mo = c2.multiselect(
            "Quy mô đội ngũ",
            [q for q in THU_TU_QM if (co["Company size"] == q).any()],
            key=f"{khoa}_qm")
        loai = c3.multiselect(
            "Loại hình", sorted(co["Company Type"].dropna().unique()),
            key=f"{khoa}_loai")
        c4, c5 = st.columns(2)
        no_ot = c4.checkbox("Chỉ công ty khai No OT", key=f"{khoa}_ot")
        vn = c5.checkbox("Chỉ công ty tại Việt Nam", key=f"{khoa}_vn")
    return {"nganh": nganh, "quy_mo": quy_mo, "loai": loai,
            "no_ot": no_ot, "vn": vn}


def ap_bo_loc(idx: np.ndarray, f: dict) -> np.ndarray:
    """Mặt nạ giữ lại những công ty thoả bộ lọc, giữ nguyên thứ tự xếp hạng."""
    sub = co.loc[idx]
    giu = np.ones(len(sub), dtype=bool)
    if f["nganh"]:
        giu &= sub["Company industry"].isin(f["nganh"]).to_numpy()
    if f["quy_mo"]:
        giu &= sub["Company size"].isin(f["quy_mo"]).to_numpy()
    if f["loai"]:
        giu &= sub["Company Type"].isin(f["loai"]).to_numpy()
    if f["no_ot"]:
        giu &= (sub["Overtime Policy"] == "No OT").to_numpy()
    if f["vn"]:
        giu &= (sub["Country"] == "Vietnam").to_numpy()
    return giu


def co_loc(f: dict) -> bool:
    return bool(f["nganh"] or f["quy_mo"] or f["loai"] or f["no_ot"] or f["vn"])


def ban_do_bam(i_goc: int | None, i_top: np.ndarray, khoa: str) -> str | None:
    """Bản đồ công ty, bấm được. Trả về tên công ty vừa bấm, hoặc None.

    Vẽ một lớp duy nhất thay vì ba lớp chồng nhau: Vega chỉ gắn được lựa chọn
    vào một lớp, và thứ tự vẽ vẫn giữ được bằng cách sắp xếp dữ liệu.
    """
    xy = load_map()
    d = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1],
                      "ten": co["Company Name"].to_numpy(),
                      "nganh": co["Company industry"].to_numpy(),
                      "quy_mo": co["Company size"].to_numpy()})
    d["vai_tro"] = "Công ty khác"
    d.loc[i_top, "vai_tro"] = "Phù hợp với bạn"
    if i_goc is not None:
        d.loc[i_goc, "vai_tro"] = "Công ty bạn chọn"

    thu_tu = ["Công ty khác", "Phù hợp với bạn", "Công ty bạn chọn"]
    # Chấm nổi bật phải nằm trên nền xám, nên xếp dữ liệu theo thứ tự đó.
    d["uu_tien"] = d["vai_tro"].map({t: i for i, t in enumerate(thu_tu)})
    d = d.sort_values("uu_tien")

    chon = alt.selection_point(name="cty", fields=["ten"], on="click",
                               clear="dblclick", empty=False)
    bieu_do = alt.Chart(d).mark_circle(stroke="#ffffff", strokeWidth=0.8).encode(
        x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
        color=alt.Color("vai_tro:N",
                        scale=alt.Scale(domain=thu_tu,
                                        range=["#d8d7d3", CAM, XANH]),
                        legend=alt.Legend(title=None, orient="top")),
        size=alt.Size("vai_tro:N",
                      scale=alt.Scale(domain=thu_tu, range=[34, 170, 330]),
                      legend=None),
        opacity=alt.Opacity("vai_tro:N",
                            scale=alt.Scale(domain=thu_tu,
                                            range=[0.4, 0.95, 1.0]),
                            legend=None),
        strokeWidth=alt.condition(chon, alt.value(2.6), alt.value(0.8)),
        tooltip=[alt.Tooltip("ten:N", title="Công ty"),
                 alt.Tooltip("nganh:N", title="Lĩnh vực"),
                 alt.Tooltip("quy_mo:N", title="Quy mô")]).add_params(chon)

    su_kien = st.altair_chart(
        bieu_do.properties(height=330), use_container_width=True,
        on_select="rerun", key=khoa)

    lua = (getattr(su_kien, "selection", None) or {}).get("cty") or []
    ten = lua[-1]["ten"] if lua else None
    khoa_cu = f"{khoa}_da_dung"
    if ten is None:
        st.session_state.pop(khoa_cu, None)
        return None
    if st.session_state.get(khoa_cu) == ten:
        return None
    st.session_state[khoa_cu] = ten
    return ten


def chi_tiet_cong_ty(ten: str, mau: str):
    """Hồ sơ một công ty, mở ra khi bấm vào bản đồ."""
    hit = co[co["Company Name"] == ten]
    if hit.empty:
        return
    r = hit.iloc[0]
    with st.container(border=True):
        st.markdown(f"#### {r['Company Name']}")
        st.caption(f"{r['Company industry']}  ·  {r['Company Type']}  ·  "
                   f"{r['Company size']}  ·  {r['Country']}  ·  "
                   f"Làm việc {r['Working days']}  ·  {r['Overtime Policy']}")
        dg = diem_danh_gia(r["id"])
        if dg is not None:
            hang_so_lieu([
                ("Điểm chung",
                 f"{dg['Overall rating']:.1f} / 5"
                 if pd.notna(dg["Overall rating"]) else "chưa có"),
                ("Tỷ lệ giới thiệu", f"{dg['ty_le_gioi_thieu']:.0f}%"),
                ("Số người đánh giá", f"{int(dg['so_review'])}"),
                ("Đánh giá gần nhất",
                 f"{pd.to_datetime(dg['review_cuoi']):%m/%Y}"),
            ], mau)
            st.write("")
            bieu_do_khia_canh(dg, mau)
        else:
            st.info("Chưa có nhân viên nào đánh giá công ty này.")
        with st.expander("Công ty này giới thiệu về họ thế nào"):
            st.write(str(r["text_raw"]))
        if isinstance(r["Href"], str) and r["Href"].startswith("http"):
            st.markdown(f"[Xem trên ITViec]({r['Href']})")


def bieu_do_khia_canh(dg: pd.Series, mau: str):
    """So điểm từng khía cạnh của một công ty với mức giữa toàn thị trường."""
    st_df = load_stats()
    trung_vi = {k: st_df[k].median() for k in KHIA}
    ss = pd.DataFrame(
        [{"khia_canh": KHIA_VI[k], "nhom": "Công ty này", "diem": dg[k]}
         for k in KHIA if pd.notna(dg[k])]
        + [{"khia_canh": KHIA_VI[k], "nhom": "Mức giữa thị trường",
            "diem": trung_vi[k]} for k in KHIA if pd.notna(dg[k])])
    if ss.empty:
        st.info("Công ty này chưa có điểm cho từng khía cạnh.")
        return
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
                range=[mau, "#c9ccd1"]),
                legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("khia_canh:N", title="Khía cạnh"),
                     alt.Tooltip("nhom:N", title=""),
                     alt.Tooltip("diem:Q", format=".2f", title="Điểm")],
        ).properties(height=270), use_container_width=True)


def so_sanh_cong_ty(ten_ds: list[str], mau: str):
    """Đặt vài công ty cạnh nhau: điểm từng khía cạnh và thông số cơ bản."""
    st_df = load_stats()
    cot = st.columns(len(ten_ds))
    for col, ten in zip(cot, ten_ds):
        r = co[co["Company Name"] == ten].iloc[0]
        dg = diem_danh_gia(r["id"])
        o_so_lieu(col, ten[:34],
                  f"{dg['Overall rating']:.1f} / 5"
                  if dg is not None and pd.notna(dg["Overall rating"])
                  else "chưa có điểm",
                  f"{dg['ty_le_gioi_thieu']:.0f}% giới thiệu · "
                  f"{int(dg['so_review'])} đánh giá" if dg is not None
                  else "chưa có ai đánh giá", mau)
    st.write("")

    hang = []
    for ten in ten_ds:
        r = co[co["Company Name"] == ten].iloc[0]
        dg = diem_danh_gia(r["id"])
        if dg is None:
            continue
        for k in KHIA:
            if pd.notna(dg[k]):
                hang.append({"cty": ten[:30], "khia_canh": KHIA_VI[k],
                             "diem": float(dg[k])})
    if not hang:
        st.info("Không công ty nào trong nhóm này có điểm từng khía cạnh để so.")
    else:
        d = pd.DataFrame(hang)
        st.markdown("**Điểm từng khía cạnh, đặt cạnh nhau**")
        st.altair_chart(
            alt.Chart(d).mark_bar(cornerRadius=3).encode(
                x=alt.X("diem:Q", title="Điểm (thang 1 tới 5)",
                        scale=alt.Scale(domain=[0, 5]),
                        axis=alt.Axis(tickCount=6)),
                y=alt.Y("khia_canh:N", sort=list(KHIA_VI.values()), title=None,
                        axis=alt.Axis(labelLimit=220, labelOverlap=False)),
                yOffset="cty:N",
                color=alt.Color("cty:N", title=None,
                                scale=alt.Scale(scheme="tableau10"),
                                legend=alt.Legend(orient="top", columns=2,
                                                  labelLimit=240)),
                tooltip=[alt.Tooltip("cty:N", title="Công ty"),
                         alt.Tooltip("khia_canh:N", title="Khía cạnh"),
                         alt.Tooltip("diem:Q", format=".2f", title="Điểm")],
            ).properties(height=90 + 54 * len(KHIA)), use_container_width=True)

    bang = co[co["Company Name"].isin(ten_ds)][
        ["Company Name", "Company industry", "Company Type", "Company size",
         "Country", "Working days", "Overtime Policy"]]
    bang = bang.merge(
        st_df[["Company Name", "Overall rating", "ty_le_gioi_thieu",
               "so_review"]], on="Company Name", how="left")
    bang = bang.rename(columns={
        "Company Name": "Công ty", "Company industry": "Lĩnh vực",
        "Company Type": "Loại hình", "Company size": "Quy mô",
        "Country": "Quốc gia", "Working days": "Ngày làm việc",
        "Overtime Policy": "Tăng ca", "Overall rating": "Điểm chung",
        "ty_le_gioi_thieu": "% giới thiệu", "so_review": "Số đánh giá"})
    st.dataframe(bang.round(2), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════ 1. Gợi ý công ty
if page in ("Tìm công ty phù hợp", "Công ty cùng nhóm với chúng tôi"):
    MAU = dai_dau_trang(page, "vai_ung_vien_anh.jpg" if LA_UNG_VIEN
                        else "vai_nha_tuyen_dung_anh.jpg")
    st.caption(f"Tra trong {meta['n_companies']} công ty công nghệ đang tuyển "
               "tại Việt Nam.")

    nhan_tab1 = ("Bắt đầu từ một công ty" if LA_UNG_VIEN
                 else "Chọn công ty của bạn")
    t1, t2, t3 = st.tabs([nhan_tab1, "Mô tả nơi bạn muốn làm",
                          "So sánh vài công ty"])

    with t1:
        ten = st.selectbox(
            "Công ty", co["Company Name"].tolist(),
            index=int(co.index[co["Company Name"].str.contains(
                "FPT Software", na=False)][:1][0])
            if co["Company Name"].str.contains("FPT Software", na=False).any()
            else 0)
        k = st.slider("Số công ty muốn xem", 3, meta["topk"], 5, key="k1")
        f1 = bo_loc_cong_ty("l1")

        i = int(co.index[co["Company Name"] == ten][0])
        goc = co.loc[i]

        with st.container(border=True):
            st.markdown(f"### {goc['Company Name']}")
            st.caption(f"{goc['Company industry']}  ·  {goc['Company Type']}  "
                       f"·  {goc['Company size']}  ·  {goc['Country']}  ·  "
                       f"Làm việc {goc['Working days']}  ·  "
                       f"{goc['Overtime Policy']}")
            with st.expander("Công ty này giới thiệu về họ thế nào"):
                st.write(goc["text_raw"])

        nn_idx, nn_score = load_neighbors()
        ca = nn_idx[i]
        giu = ap_bo_loc(ca, f1)
        top = ca[giu][:k]
        diem_top = nn_score[i][giu][:k]

        if not len(top):
            st.warning(
                f"Trong {len(ca)} công ty gần nhất với {goc['Company Name']}, "
                "không công ty nào thoả bộ lọc. Hãy bỏ bớt tiêu chí."
            )
        else:
            if co_loc(f1):
                st.caption(
                    f"Bộ lọc giữ lại {int(giu.sum())} trong {len(ca)} công ty "
                    f"gần nhất, đang hiện {len(top)} công ty."
                )
            st.subheader(f"{len(top)} công ty gần giống nhất")

            c1, c2 = st.columns([2, 3])
            with c1:
                st.altair_chart(bieu_do_do_hop(co.loc[top], diem_top, MAU),
                                use_container_width=True)
            with c2:
                bam = ban_do_bam(i, top, "bd_t1")
                if bam:
                    st.session_state["cty_mo_t1"] = bam
                st.caption(
                    "Mỗi chấm là một công ty. Hai công ty càng gần nhau thì "
                    "phần giới thiệu của họ càng giống nhau. **Bấm vào một "
                    "chấm** để mở hồ sơ công ty đó."
                )

            mo = st.session_state.get("cty_mo_t1")
            if mo:
                chi_tiet_cong_ty(mo, MAU)

            st.caption("Chấm xanh lá trước tên nghĩa là cùng lĩnh vực với "
                       "công ty bạn chọn. Lĩnh vực không được đưa vào hệ "
                       "thống, nên đây là cách bạn tự kiểm chứng chất lượng "
                       "gợi ý.")
            the_cong_ty(co.loc[top], diem_top, goc)

    with t2:
        st.write(
            "Tả bằng lời nơi bạn muốn làm việc, tiếng Việt hay tiếng Anh đều "
            "được. Ứng dụng tự nhận ra ngôn ngữ bạn dùng."
            if LA_UNG_VIEN else
            "Tả định vị hoặc lĩnh vực của công ty bạn để tìm nhóm cạnh tranh."
        )
        goi_y_mau = {
            "Fintech và thanh toán":
                "Công ty fintech phát triển sản phẩm thanh toán, môi trường "
                "trẻ, có cơ hội học hỏi về blockchain và dữ liệu lớn",
            "Gia công phần mềm cho Nhật":
                "Công ty outsourcing làm dự án cho khách hàng Nhật Bản, có "
                "đào tạo tiếng Nhật, quy trình bài bản",
            "Sản phẩm AI và dữ liệu":
                "Product company làm về trí tuệ nhân tạo và xử lý dữ liệu "
                "lớn, đội ngũ kỹ sư mạnh, văn hoá cởi mở",
        }
        chon = st.radio("Chọn một mẫu có sẵn hoặc tự viết",
                        list(goi_y_mau) + ["Tự viết"], horizontal=True)
        mac_dinh = goi_y_mau.get(chon, "")
        truy_van = st.text_area("Mô tả của bạn", mac_dinh, height=110)
        k2 = st.slider("Số công ty muốn xem", 3, 20, 5, key="k2")
        f2 = bo_loc_cong_ty("l2")

        if st.button("Tìm công ty", type="primary", use_container_width=True):
            st.session_state["tv_tim"] = truy_van
            st.session_state.pop("cty_mo_t2", None)

        if st.session_state.get("tv_tim"):
            sach, lang = preprocess(st.session_state["tv_tim"])
            if not sach.strip():
                st.warning("Chưa có nội dung để tìm. Hãy viết vài câu mô tả "
                           "nơi bạn muốn làm việc.")
            else:
                s = load_search()
                sim = cosine_similarity(s["vectorizer"].transform([sach]),
                                        s["matrix"]).ravel()
                xep = np.argsort(-sim)
                giu = ap_bo_loc(xep, f2)
                top = xep[giu][:k2]

                if not len(top):
                    st.warning("Không công ty nào thoả bộ lọc. Hãy bỏ bớt "
                               "tiêu chí.")
                elif sim[top[0]] < 0.05:
                    st.warning(
                        "Không tìm được công ty nào thật sự khớp. Thử viết "
                        "dài hơn, hoặc dùng những từ hay xuất hiện trong mô "
                        "tả tuyển dụng như tên công nghệ, lĩnh vực, quy mô "
                        "đội ngũ."
                    )
                else:
                    st.caption(
                        "Nhận ra bạn đang viết bằng "
                        f"**{'tiếng Việt' if lang == 'vi' else 'tiếng Anh'}**."
                        + (f" Bộ lọc giữ lại {int(giu.sum())} công ty."
                           if co_loc(f2) else "")
                    )
                    c1, c2 = st.columns([2, 3])
                    with c1:
                        st.altair_chart(
                            bieu_do_do_hop(co.loc[top], sim[top], MAU),
                            use_container_width=True)
                    with c2:
                        bam = ban_do_bam(None, top, "bd_t2")
                        if bam:
                            st.session_state["cty_mo_t2"] = bam
                        st.caption("Vị trí các công ty phù hợp trên bản đồ. "
                                   "**Bấm vào một chấm** để mở hồ sơ công ty.")
                    mo = st.session_state.get("cty_mo_t2")
                    if mo:
                        chi_tiet_cong_ty(mo, MAU)
                    the_cong_ty(co.loc[top], sim[top], None)

    with t3:
        st.write(
            "Chọn hai tới bốn công ty để đặt điểm đánh giá và thông số của họ "
            "cạnh nhau. Chỉ những công ty đã có người đánh giá mới có điểm "
            "từng khía cạnh."
        )
        ten_ss = st.multiselect(
            "Công ty muốn so", co["Company Name"].tolist(),
            default=co["Company Name"].tolist()[:2], max_selections=4,
            key="ss_cty")
        if len(ten_ss) < 2:
            st.info("Hãy chọn ít nhất hai công ty.")
        else:
            so_sanh_cong_ty(ten_ss, MAU)

# ═══════════════════════════════════ 2. Đọc hiểu đánh giá
elif page in ("Đọc hiểu một đánh giá", "Phân tích phản hồi nhân viên"):
    MAU = dai_dau_trang(page, "vai_ung_vien_anh.jpg" if LA_UNG_VIEN
                        else "vai_nha_tuyen_dung_anh.jpg")
    st.caption(
        f"Mô hình học từ {m2['n_train'] + m2['n_test']:,} đánh giá thật của "
        "nhân viên."
    )

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
        st.session_state["dg_chay"] = True

    if st.session_state.get("dg_chay"):
        tho = f"{tieu_de} {thich} {gop_y}".strip()
        sach, lang = preprocess(tho)

        if not sach.strip():
            st.warning("Chưa có nội dung đánh giá để phân tích.")
        else:
            X = pd.DataFrame([{**diem, "text_clean": sach}])
            p_yes = float(load_classifier().predict_proba(X)[0, 1])
            la_yes = p_yes >= 0.5
            mau_kq = LUC if la_yes else CAM
            icon = "👍" if la_yes else "👎"
            tieu = ("Nghiêng về SẼ giới thiệu công ty"
                    if la_yes else "Nghiêng về KHÔNG giới thiệu công ty")
            chac_chan = ("rất rõ ràng" if abs(p_yes - 0.5) > 0.35
                         else "khá rõ" if abs(p_yes - 0.5) > 0.2
                         else "không chắc chắn lắm")

            st.divider()
            tk1, tk2 = st.tabs(["Kết quả", "Điều gì quyết định kết quả"])

            # ─────────────────────────────────── Tab 1: kết quả
            with tk1:
                st.markdown(
                    f"<div style='padding:20px;border-radius:12px;"
                    f"background:{mau_kq}18;border-left:6px solid {mau_kq}'>"
                    f"<div style='font-size:25px;font-weight:700;"
                    f"color:{mau_kq}'>{icon} {tieu}</div>"
                    f"<div style='color:{XAM};margin-top:8px'>Mức độ: "
                    f"<b>{chac_chan}</b>. Khả năng người viết sẽ giới thiệu "
                    f"công ty là <b>{p_yes * 100:.0f}%</b>.</div></div>",
                    unsafe_allow_html=True)

                st.write("")
                hang_so_lieu([
                    ("Khả năng sẽ giới thiệu", f"{p_yes * 100:.0f}%",
                     "theo mô hình"),
                    ("Số sao trung bình",
                     f"{np.mean(list(diem.values())):.1f} / 5",
                     "trên sáu mục người viết chấm"),
                    ("Ngôn ngữ",
                     "Tiếng Việt" if lang == "vi" else "Tiếng Anh",
                     "ứng dụng tự nhận ra"),
                ], mau_kq)
                st.write("")

                thanh = pd.DataFrame(
                    {"nhan": ["Không giới thiệu", "Sẽ giới thiệu"],
                     "p": [1 - p_yes, p_yes]})
                nen = alt.Chart(thanh).encode(
                    y=alt.Y("nhan:N",
                            sort=["Không giới thiệu", "Sẽ giới thiệu"],
                            title=None,
                            axis=alt.Axis(labelFontSize=13, labelPadding=10)))
                cot = nen.mark_bar(cornerRadius=4, size=34).encode(
                    x=alt.X("p:Q", scale=alt.Scale(domain=[0, 1]),
                            axis=alt.Axis(format="%"), title="Khả năng"),
                    color=alt.Color("nhan:N", scale=alt.Scale(
                        domain=["Không giới thiệu", "Sẽ giới thiệu"],
                        range=[CAM, LUC]), legend=None))
                # Nhãn số đặt cạnh đầu cột. Với khả năng rất nhỏ thì cột gần
                # như vô hình, lúc đó chỉ còn con số cho biết giá trị thật.
                chu = nen.mark_text(align="left", dx=7, fontSize=12,
                                    color=XAM).encode(
                    x=alt.X("p:Q", scale=alt.Scale(domain=[0, 1])),
                    text=alt.Text("p:Q", format=".0%"))
                st.altair_chart((cot + chu).properties(height=150),
                                use_container_width=True)

                st.info(
                    "Đây là dự đoán, không phải sự thật. Trong các đánh giá "
                    "thực tế thuộc nhóm không giới thiệu, ứng dụng nhận ra "
                    f"được khoảng {m2['Recall_No'] * 100:.0f}%. Khi ứng dụng "
                    "báo không giới thiệu thì đúng khoảng "
                    f"{m2['Precision_No'] * 100:.0f}%."
                )

            # ─────────────────────── Tab 2: số sao hay nội dung quyết định
            with tk2:
                st.write(
                    "Giữ nguyên nội dung bạn vừa dán, chỉ thay đổi số sao, "
                    "rồi xem kết quả đổi ra sao. Chọn thay đổi tất cả các mục "
                    "cùng lúc, hoặc chỉ một mục để biết mục nào có sức nặng."
                )
                muc_quet = st.radio(
                    "Mục muốn thử", ["Tất cả các mục"] + NUMERIC,
                    format_func=lambda c: NUMERIC_VI.get(c, c),
                    horizontal=False, key="dg_muc_quet")

                quet = []
                for d in [1.0, 2.0, 3.0, 4.0, 5.0]:
                    if muc_quet == "Tất cả các mục":
                        bo = {c: d for c in NUMERIC}
                    else:
                        bo = {**diem, muc_quet: d}
                    Xd = pd.DataFrame([{**bo, "text_clean": sach}])
                    quet.append({"diem": d, "p": float(
                        load_classifier().predict_proba(Xd)[0, 1])})
                qdf = pd.DataFrame(quet)

                nhan_truc = ("Nếu người viết chấm tất cả bằng"
                             if muc_quet == "Tất cả các mục"
                             else f"Nếu mục {NUMERIC_VI[muc_quet].lower()} "
                                  "được chấm bằng")
                duong = alt.Chart(qdf).mark_line(
                    point=alt.OverlayMarkDef(size=80, color=MAU), color=MAU,
                    strokeWidth=3).encode(
                    x=alt.X("diem:Q", title=nhan_truc,
                            scale=alt.Scale(domain=[0.8, 5.2]),
                            axis=alt.Axis(values=[1, 2, 3, 4, 5])),
                    y=alt.Y("p:Q", title="Khả năng sẽ giới thiệu",
                            scale=alt.Scale(domain=[0, 1]),
                            axis=alt.Axis(format="%")),
                    tooltip=[alt.Tooltip("diem:Q", title="Số sao"),
                             alt.Tooltip("p:Q", format=".0%",
                                         title="Khả năng")])
                moc = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
                    color=DO, strokeDash=[6, 4], size=2).encode(y="y:Q")
                st.altair_chart((duong + moc).properties(height=300),
                                use_container_width=True)

                bien_do = qdf["p"].max() - qdf["p"].min()
                hang_so_lieu([
                    ("Thấp nhất", f"{qdf['p'].min() * 100:.0f}%",
                     "khi chấm 1 sao"),
                    ("Cao nhất", f"{qdf['p'].max() * 100:.0f}%",
                     "khi chấm 5 sao"),
                    ("Sức nặng của mục này", f"{bien_do * 100:.0f} điểm %",
                     "khoảng cách giữa hai đầu"),
                ], MAU)
                st.write("")
                st.write(
                    "Khi chấm 5 sao thì gần như luôn là sẽ giới thiệu, chấm 1 "
                    "sao thì gần như luôn ngược lại, bất kể viết gì. **Vùng "
                    "quanh 3 sao mới là nơi nội dung quyết định**, và đó cũng "
                    "là lúc công cụ này hữu ích nhất. Đổi sang từng mục riêng "
                    "để xem mục nào kéo kết quả mạnh nhất."
                )


# ═══════════════════════════════════ 3. Toàn cảnh thị trường
elif page == "Toàn cảnh thị trường IT":
    MAU = dai_dau_trang(page, "itviec_minhhoa.jpg")

    hang_so_lieu([
        ("Công ty", f"{len(co)}", "đang tuyển dụng"),
        ("Lĩnh vực", f"{co['Company industry'].nunique()}", "ngành nghề"),
        ("Công ty Việt Nam", f"{(co['Country'] == 'Vietnam').sum()}",
         "phần còn lại là công ty nước ngoài"),
        ("Khai báo No OT", f"{(co['Overtime Policy'] == 'No OT').sum()}",
         "do công ty tự khai"),
    ], MAU)
    st.write("")
    st.caption(
        "Lưu ý: chính sách tăng ca là do công ty tự khai khi đăng tin, nên "
        "nên xem như lời giới thiệu chứ không phải thực tế đã được kiểm chứng."
    )

    st_df = load_stats()
    tt1, tt2, tt3, tt4, tt5 = st.tabs(
        ["Bức tranh chung", "Điểm và số đánh giá", "Tra một công ty",
         "So sánh công ty", "Theo thời gian"])

    # ──────────────────────────────── Tab 1: bức tranh chung
    with tt1:
        st.caption("**Bấm vào một lĩnh vực** để lọc nhanh biểu đồ ở thẻ "
                   "Điểm và số đánh giá. Bấm đúp vào vùng trống để bỏ chọn.")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Lĩnh vực nào nhiều công ty nhất")
            vc = (co["Company industry"].value_counts().head(12)
                  .rename_axis("linh_vuc").reset_index(name="so_cty"))
            chon_nganh = alt.selection_point(
                name="nganh", fields=["linh_vuc"], on="click",
                clear="dblclick", empty=False)
            bd = alt.Chart(vc).mark_bar(cornerRadius=3).encode(
                x=alt.X("so_cty:Q", title="Số công ty",
                        axis=alt.Axis(tickCount=6)),
                y=alt.Y("linh_vuc:N", sort="-x", title=None,
                        axis=alt.Axis(labelLimit=250, labelOverlap=False)),
                color=alt.condition(chon_nganh, alt.value(CAM),
                                    alt.value(MAU)),
                tooltip=[alt.Tooltip("linh_vuc:N", title="Lĩnh vực"),
                         alt.Tooltip("so_cty:Q", title="Số công ty")],
            ).add_params(chon_nganh)
            sk = st.altair_chart(bd.properties(height=330),
                                 use_container_width=True,
                                 on_select="rerun", key="tt_bd_nganh")
            lua = (getattr(sk, "selection", None) or {}).get("nganh") or []
            if lua:
                st.session_state["tt_nganh_nhanh"] = lua[-1]["linh_vuc"]
            else:
                st.session_state.pop("tt_nganh_nhanh", None)

        with c2:
            st.subheader("Quy mô đội ngũ")
            qm = (co["Company size"].value_counts()
                  .rename_axis("quy_mo").reset_index(name="so_cty"))
            qm["quy_mo"] = pd.Categorical(qm["quy_mo"], THU_TU_QM,
                                          ordered=True)
            qm = qm.dropna(subset=["quy_mo"]).sort_values("quy_mo")
            cot_qm = alt.Chart(qm).mark_bar(cornerRadius=3, color=MAU).encode(
                x=alt.X("so_cty:Q", title="Số công ty",
                        axis=alt.Axis(tickCount=6)),
                y=alt.Y("quy_mo:N", sort=THU_TU_QM, title=None,
                        axis=alt.Axis(labelLimit=200, labelOverlap=False)),
                tooltip=[alt.Tooltip("quy_mo:N", title="Quy mô"),
                         alt.Tooltip("so_cty:Q", title="Số công ty")])
            nhan_qm = alt.Chart(qm).mark_text(
                align="left", dx=6, fontSize=12, fontWeight="bold",
                color="#0b0b0b").encode(
                x="so_cty:Q", y=alt.Y("quy_mo:N", sort=THU_TU_QM),
                text="so_cty:Q")
            st.altair_chart((cot_qm + nhan_qm).properties(height=330),
                            use_container_width=True)
            st.caption("Phần lớn là công ty nhỏ và vừa. Nếu bạn muốn môi "
                       "trường đông người thì nhóm trên 1000 nhân sự chỉ có "
                       "vài chục lựa chọn.")

        st.divider()
        st.subheader("Người ta viết đánh giá bằng ngôn ngữ nào")
        # Cột trái rộng hơn cột phải: nhãn "Nhân viên đánh giá" và tiêu đề
        # trục đều dài, cột hẹp sẽ cắt mất chữ.
        c1, c2 = st.columns([3, 2])
        with c1:
            ngon_ngu = pd.DataFrame({
                "nguon": ["Công ty tự giới thiệu", "Nhân viên đánh giá"],
                "vi": [31.2, 88.9]})
            nen_nn = alt.Chart(ngon_ngu).encode(
                y=alt.Y("nguon:N", sort=None, title=None,
                        axis=alt.Axis(labelLimit=220, labelOverlap=False)))
            cot_nn = nen_nn.mark_bar(cornerRadius=3, color=MAU,
                                     size=30).encode(
                x=alt.X("vi:Q", title="Viết bằng tiếng Việt (%)",
                        scale=alt.Scale(domain=[0, 100])),
                tooltip=[alt.Tooltip("nguon:N", title="Nguồn"),
                         alt.Tooltip("vi:Q", format=".1f",
                                     title="% tiếng Việt")])
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

    # ──────────────────────── Tab 2: điểm và số người đánh giá
    with tt2:
        st.subheader("Điểm cao có đi kèm nhiều người đánh giá không?")
        nhanh = st.session_state.get("tt_nganh_nhanh")
        if nhanh:
            st.info(f"Đang lọc nhanh theo lĩnh vực **{nhanh}**, chọn từ thẻ "
                    "Bức tranh chung. Bấm đúp vào biểu đồ bên đó để bỏ.")
        nganh_chon = st.multiselect(
            "Lọc theo lĩnh vực, để trống là xem tất cả",
            sorted(st_df["Company industry"].dropna().unique()),
            key="tt_nganh")
        loc = st_df
        if nganh_chon:
            loc = loc[loc["Company industry"].isin(nganh_chon)]
        if nhanh:
            loc = loc[loc["Company industry"] == nhanh]
        loc = loc.dropna(subset=["Overall rating"])

        if loc.empty:
            st.warning("Không công ty nào thoả bộ lọc hiện tại.")
        else:
            hang_so_lieu([
                ("Công ty đang hiện", f"{len(loc)}", "có điểm đánh giá"),
                ("Điểm chung giữa", f"{loc['Overall rating'].median():.2f}",
                 "trên thang 5"),
                ("Tỷ lệ giới thiệu giữa",
                 f"{loc['ty_le_gioi_thieu'].median():.0f}%", "mức trung vị"),
                ("Số đánh giá giữa", f"{loc['so_review'].median():.0f}",
                 "mỗi công ty"),
            ], MAU)
            st.write("")
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
                             alt.Tooltip("Company industry:N",
                                         title="Lĩnh vực"),
                             alt.Tooltip("so_review:Q", title="Số đánh giá"),
                             alt.Tooltip("Overall rating:Q", format=".2f",
                                         title="Điểm"),
                             alt.Tooltip("ty_le_gioi_thieu:Q", format=".0f",
                                         title="% giới thiệu")],
                ).properties(height=400).add_params(zoom_giu_shift()),
                use_container_width=True)
            st.caption(
                f"Đang hiện {len(loc)} công ty. Trục ngang dùng thang log vì "
                "số đánh giá chênh nhau rất lớn, từ 3 tới hơn hai nghìn. Rê "
                "chuột lên từng chấm để xem tên. Giữ Shift rồi cuộn để phóng "
                "to, kéo để di chuyển. **Các chấm bên trái là công ty có rất "
                "ít đánh giá**, điểm của họ kém tin cậy hơn dù cao hay thấp."
            )

    # ──────────────────────────────── Tab 3: tra một công ty
    with tt3:
        st.subheader("Tra điểm đánh giá của một công ty")
        ten_tra = st.selectbox(
            f"Chọn công ty, {len(st_df)} công ty đã có người đánh giá",
            st_df.sort_values("so_review",
                              ascending=False)["Company Name"].tolist(),
            key="tt_tra")
        r = st_df[st_df["Company Name"] == ten_tra].iloc[0]

        hang_so_lieu([
            ("Điểm chung",
             f"{r['Overall rating']:.1f} / 5"
             if pd.notna(r["Overall rating"]) else "chưa có", "trên thang 5"),
            ("Tỷ lệ sẽ giới thiệu", f"{r['ty_le_gioi_thieu']:.0f}%",
             "trong số người đã đánh giá"),
            ("Số người đánh giá", f"{int(r['so_review'])}", "tới nay"),
            ("Đánh giá gần nhất",
             f"{pd.to_datetime(r['review_cuoi']):%m/%Y}", "tháng và năm"),
        ], MAU)
        st.write("")

        if r["so_review"] < 10:
            st.warning(
                f"Chỉ {int(r['so_review'])} người đánh giá công ty này. Với "
                "số lượng ít như vậy, một ý kiến cực đoan cũng đủ kéo lệch "
                "điểm trung bình. Hãy đọc kỹ từng đánh giá thay vì nhìn con "
                "số tổng."
            )

        # So điểm từng khía cạnh với mức giữa của toàn thị trường, để người
        # xem biết công ty này mạnh yếu ở đâu chứ không chỉ thấy một con số.
        bieu_do_khia_canh(r, MAU)

    # ──────────────────────────────── Tab 4: so sánh công ty
    with tt4:
        st.write("Chọn hai tới bốn công ty đã có người đánh giá để đặt điểm "
                 "của họ cạnh nhau.")
        ten_ss2 = st.multiselect(
            "Công ty muốn so",
            st_df.sort_values("so_review",
                              ascending=False)["Company Name"].tolist(),
            default=st_df.sort_values(
                "so_review", ascending=False)["Company Name"].tolist()[:2],
            max_selections=4, key="tt_ss")
        if len(ten_ss2) < 2:
            st.info("Hãy chọn ít nhất hai công ty.")
        else:
            so_sanh_cong_ty(ten_ss2, MAU)

    # ──────────────────────────────── Tab 5: theo thời gian
    with tt5:
        st.subheader("Dữ liệu này mới tới đâu")
        th = load_monthly()
        th["nam"] = pd.to_datetime(th["Cmt_date"]).dt.year
        nam_min, nam_max = int(th["nam"].min()), int(th["nam"].max())
        khoang = st.slider("Khoảng thời gian", nam_min, nam_max,
                           (nam_min, nam_max), key="tt_nam")
        tv = th[(th["nam"] >= khoang[0]) & (th["nam"] <= khoang[1])]

        hang_so_lieu([
            ("Đánh giá trong khoảng", f"{int(tv['so_review'].sum()):,}",
             f"từ {khoang[0]} tới {khoang[1]}"),
            ("Tỷ lệ giới thiệu trung bình",
             f"{tv['ty_le_gioi_thieu'].mean():.0f}%", "tính theo tháng"),
            ("Tháng nhiều đánh giá nhất",
             f"{pd.to_datetime(tv.loc[tv['so_review'].idxmax(), 'Cmt_date']):%m/%Y}"
             if len(tv) else "chưa có",
             f"{int(tv['so_review'].max()):,} đánh giá" if len(tv) else ""),
        ], MAU)
        st.write("")

        chon_tg = st.radio("Xem theo", ["Số đánh giá mỗi tháng",
                                        "Tỷ lệ giới thiệu mỗi tháng"],
                           horizontal=True, key="tt_xem")
        cot_tg = ("so_review" if chon_tg == "Số đánh giá mỗi tháng"
                  else "ty_le_gioi_thieu")
        tieu_tg = ("Số đánh giá mỗi tháng"
                   if cot_tg == "so_review" else "Tỷ lệ giới thiệu (%)")
        st.altair_chart(
            alt.Chart(tv).mark_area(
                line={"color": MAU}, opacity=0.25, color=MAU).encode(
                x=alt.X("Cmt_date:T", title="Thời gian"),
                y=alt.Y(f"{cot_tg}:Q", title=tieu_tg,
                        scale=alt.Scale(zero=cot_tg == "so_review")),
                tooltip=[alt.Tooltip("Cmt_date:T", title="Tháng",
                                     format="%m/%Y"),
                         alt.Tooltip("so_review:Q", title="Số đánh giá"),
                         alt.Tooltip("ty_le_gioi_thieu:Q", format=".0f",
                                     title="% giới thiệu")],
            ).properties(height=320).add_params(zoom_giu_shift()),
            use_container_width=True)
        st.caption("Một công ty có thể đã thay đổi nhiều so với lúc người ta "
                   "viết đánh giá, nhất là các năm đầu.")


# ═══════════════════════════════════════════════ 4. Về ứng dụng
else:
    MAU = dai_dau_trang("Về ứng dụng")

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
    hang_so_lieu([
        ("Công ty", f"{meta['n_companies']}",
         "có phần giới thiệu đủ dài để so sánh"),
        ("Đánh giá của nhân viên", f"{m2['n_train'] + m2['n_test']:,}",
         "thu thập từ 2016 tới 2025"),
        ("Độ chính xác khi nhận ra người không hài lòng",
         f"{m2['Precision_No'] * 100:.0f}%",
         "trong số lần báo không giới thiệu"),
    ], MAU)

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
        hang_so_lieu([
            ("P@5 lĩnh vực", f"{p_tot_nhat * 100:.1f}%",
             "mốc ngẫu nhiên "
             f"{meta['yeucau1']['baseline']['Company industry'] * 100:.1f}%"),
            ("F1 lớp thiểu số", f"{m2['F1_No']:.3f}", "đoán bừa cho 0,000"),
            ("ROC-AUC", f"{m2['ROC_AUC']:.3f}", "trên tập kiểm tra"),
            ("Accuracy", f"{m2['Accuracy']:.3f}",
             f"đoán bừa đạt {m2['baseline_accuracy'] * 100:.1f}%"),
        ], MAU)
        st.write("")

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
