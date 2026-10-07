"""Vẽ minh hoạ chủ đề cho hai ứng dụng.

Tự vẽ bằng matplotlib thay vì tải ảnh có sẵn, vừa tránh vấn đề bản quyền
vừa giữ đúng bảng màu của ứng dụng.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

mpl.use("Agg")

BASE = Path("/home/dang.lu/Personal/Final ML")
T1 = BASE / "GUI_Project1" / "figures"
P2 = BASE / "GUI_Project2" / "figures"
for d in (T1, P2):
    d.mkdir(parents=True, exist_ok=True)

XANH = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CAM, LUC, VANG = "#eb6834", "#1baf7a", "#f5c24c"
TRANG, MUC, XAM = "#ffffff", "#0b0b0b", "#52514e"


def khung(w, h, nen):
    fig = plt.figure(figsize=(w, h), dpi=100)
    fig.patch.set_facecolor(nen)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100 * w / h)
    ax.set_ylim(0, 100)
    ax.set_axis_off()
    ax.set_aspect("equal")
    return fig, ax


def luu(fig, path, nen):
    fig.savefig(path, facecolor=nen)
    plt.close(fig)
    print(f"  {path.relative_to(BASE)}  {path.stat().st_size / 1024:.0f} KB")


# ═══════════════════════════════════════════════ Chi tiết dùng lại
def cua_so(ax, x, y, w, h, mau=VANG, vien="#ffffff"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=mau, edgecolor=vien,
                           linewidth=1.1, zorder=5))


def nha_pho(ax, x, nen_y, rong, cao, than, mai, n_tang=2, cua_chinh=True):
    """Một căn nhà phố: thân hình chữ nhật, mái dốc, cửa sổ xếp lưới."""
    ax.add_patch(Rectangle((x, nen_y), rong, cao, facecolor=than,
                           edgecolor="none", zorder=3))
    ax.add_patch(Polygon([[x - rong * 0.09, nen_y + cao],
                          [x + rong / 2, nen_y + cao + rong * 0.38],
                          [x + rong * 1.09, nen_y + cao]],
                         closed=True, facecolor=mai, edgecolor="none", zorder=4))
    cs_w, cs_h = rong * 0.21, cao * 0.13
    for t in range(n_tang):
        y = nen_y + cao * (0.74 - 0.27 * t)
        for c in range(2):
            cua_so(ax, x + rong * (0.21 + 0.42 * c), y, cs_w, cs_h)
    if cua_chinh:
        ax.add_patch(Rectangle((x + rong * 0.38, nen_y), rong * 0.24, cao * 0.22,
                               facecolor="#8a5a3b", edgecolor="none", zorder=5))


def toa_nha(ax, x, nen_y, rong, cao, than, n_cot=4, n_hang=7):
    """Toà nhà văn phòng: khối chữ nhật với lưới cửa kính."""
    ax.add_patch(Rectangle((x, nen_y), rong, cao, facecolor=than,
                           edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((x - rong * 0.04, nen_y + cao), rong * 1.08,
                           cao * 0.035, facecolor="#ffffff", alpha=0.25,
                           edgecolor="none", zorder=4))
    kw = rong * 0.72 / n_cot
    kh = cao * 0.72 / n_hang
    rng = np.random.default_rng(int(x * 7) + n_hang)
    for i in range(n_cot):
        for j in range(n_hang):
            sang = rng.random() < 0.34
            cua_so(ax,
                   x + rong * 0.14 + i * kw + kw * 0.14,
                   nen_y + cao * 0.12 + j * kh + kh * 0.16,
                   kw * 0.72, kh * 0.62,
                   mau=VANG if sang else "#bcd6f2", vien="none")


def cay(ax, x, nen_y, cao, mau=LUC):
    ax.add_patch(Rectangle((x - cao * 0.045, nen_y), cao * 0.09, cao * 0.34,
                           facecolor="#7a5a3f", edgecolor="none", zorder=4))
    for dy, r in [(0.46, 0.21), (0.62, 0.17), (0.76, 0.12)]:
        ax.add_patch(Circle((x, nen_y + cao * dy), cao * r, facecolor=mau,
                            edgecolor="none", zorder=5))


def may(ax, x, y, s, mau="#ffffff", alpha=0.7):
    for dx, dy, r in [(0, 0, 1.0), (0.85, 0.12, 0.74), (-0.8, 0.08, 0.66),
                      (0.3, 0.42, 0.62)]:
        ax.add_patch(Circle((x + dx * s, y + dy * s), r * s, facecolor=mau,
                            alpha=alpha, edgecolor="none", zorder=2))


# ═══════════════════════════════════════ Ảnh đầu trang Topic 1
def hero_nha():
    nen = "#dceaf9"
    fig, ax = khung(16, 4.2, nen)
    W = ax.get_xlim()[1]

    # Mặt trời và mây đặt ở khoảng trời còn trống phía trên nóc nhà cao nhất.
    ax.add_patch(Circle((205, 92), 7.5, facecolor="#ffd05e",
                        edgecolor="none", zorder=1))
    may(ax, 60, 92, 5.5, alpha=0.85)
    may(ax, 262, 93, 4.6, alpha=0.85)
    may(ax, 340, 95, 4.0, alpha=0.85)

    ax.add_patch(Rectangle((0, 0), W, 26, facecolor="#cfe0d2",
                           edgecolor="none", zorder=1))
    ax.add_patch(Rectangle((0, 0), W, 13, facecolor="#9aa4a0",
                           edgecolor="none", zorder=2))
    for i in range(16):
        ax.add_patch(Rectangle((8 + i * 24, 6), 11, 1.6, facecolor="#e6e9e8",
                               edgecolor="none", zorder=3))

    # Dãy nhà phố cao thấp xen kẽ, màu chuyển dần cho có chiều sâu
    # Chiều cao đã hạ để đỉnh mái dốc cao nhất vẫn nằm dưới mép trên khung.
    cau_hinh = [
        (18, 40, 39, XANH[2], XANH[5], 2), (62, 34, 32, XANH[1], XANH[4], 2),
        (100, 44, 45, XANH[3], XANH[6], 3), (148, 36, 35, XANH[2], XANH[5], 2),
        (188, 42, 42, XANH[1], XANH[4], 2), (234, 33, 30, XANH[3], XANH[6], 2),
        (271, 39, 37, XANH[2], XANH[5], 2), (314, 45, 46, XANH[1], XANH[4], 3),
    ]
    for x, rong, cao, than, mai, nt in cau_hinh:
        nha_pho(ax, x, 26, rong, cao, than, mai, nt)

    for x in (8, 93, 182, 262, 352):
        cay(ax, x, 26, 30)

    luu(fig, T1 / "hero_nha.png", nen)


# ═══════════════════════════════════ Ảnh đầu trang Project 2
def hero_vanphong():
    nen = "#dfe8f6"
    fig, ax = khung(16, 4.2, nen)
    W = ax.get_xlim()[1]

    may(ax, 42, 93, 5.2, alpha=0.8)
    may(ax, 190, 95, 4.4, alpha=0.8)
    may(ax, 288, 94, 4.8, alpha=0.8)

    ax.add_patch(Rectangle((0, 0), W, 17, facecolor="#c3cedd",
                           edgecolor="none", zorder=1))

    # Chiều cao đã hạ để nóc toà cao nhất vẫn nằm dưới mép trên khung.
    thap = [
        (14, 34, 39, XANH[2], 3, 6), (54, 28, 55, XANH[4], 3, 8),
        (88, 40, 45, XANH[1], 4, 7), (134, 30, 62, XANH[5], 3, 9),
        (170, 36, 36, XANH[2], 4, 5), (212, 32, 56, XANH[3], 3, 8),
        (250, 42, 48, XANH[1], 4, 7), (298, 28, 67, XANH[6], 3, 10),
        (332, 36, 42, XANH[2], 4, 6),
    ]
    for x, rong, cao, than, nc, nh in thap:
        toa_nha(ax, x, 17, rong, cao, than, nc, nh)

    for x in (6, 128, 204, 290, 378):
        cay(ax, x, 17, 24)

    luu(fig, P2 / "hero_vanphong.png", nen)


# ═══════════════════════════════════════════ Thẻ chọn vai trò
def the(path, ve, nen="#f2f6fc"):
    fig, ax = khung(4.4, 3.0, nen)
    ve(ax, ax.get_xlim()[1])
    luu(fig, path, nen)


def ve_nguoi_ban(ax, W):
    """Nhà kèm bảng rao bán."""
    nha_pho(ax, W * 0.17, 16, W * 0.44, 46, XANH[2], XANH[5], 2)
    ax.add_patch(Rectangle((0, 0), W, 16, facecolor="#cfe0d2",
                           edgecolor="none", zorder=1))
    # Bảng cắm trước nhà
    ax.add_patch(Rectangle((W * 0.70, 16), 2.4, 34, facecolor="#8a5a3b",
                           edgecolor="none", zorder=6))
    ax.add_patch(FancyBboxPatch((W * 0.60, 48), W * 0.32, 22,
                                boxstyle="round,pad=1.4,rounding_size=3",
                                facecolor=CAM, edgecolor="none", zorder=7))
    ax.text(W * 0.60 + W * 0.16, 59, "BÁN", fontsize=15, fontweight="bold",
            color=TRANG, ha="center", va="center", zorder=8)
    cay(ax, W * 0.07, 16, 26)


def ve_nguoi_mua(ax, W):
    """Nhà kèm kính lúp soi giá."""
    nha_pho(ax, W * 0.10, 16, W * 0.44, 46, XANH[2], XANH[5], 2)
    ax.add_patch(Rectangle((0, 0), W, 16, facecolor="#cfe0d2",
                           edgecolor="none", zorder=1))
    cx, cy, r = W * 0.70, 52, 15
    ax.add_patch(Circle((cx, cy), r, facecolor="#ffffff", alpha=0.92,
                        edgecolor=XANH[6], linewidth=3.4, zorder=7))
    ax.plot([cx + r * 0.72, cx + r * 1.65], [cy - r * 0.72, cy - r * 1.65],
            color=XANH[6], linewidth=5.4, solid_capstyle="round", zorder=7)
    ax.text(cx, cy, "₫", fontsize=20, fontweight="bold", color=CAM,
            ha="center", va="center", zorder=8)
    cay(ax, W * 0.90, 16, 24)


def ve_ung_vien(ax, W):
    """Cặp tài liệu kèm kính lúp, gợi ý việc tìm nơi làm việc."""
    ax.add_patch(Rectangle((0, 0), W, 14, facecolor="#c3cedd",
                           edgecolor="none", zorder=1))
    bx, by, bw, bh = W * 0.16, 22, W * 0.44, 34
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
                                boxstyle="round,pad=0,rounding_size=3",
                                facecolor=XANH[4], edgecolor="none", zorder=4))
    ax.add_patch(Rectangle((bx + bw * 0.34, by + bh), bw * 0.32, 7,
                           facecolor=XANH[6], edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((bx, by + bh * 0.42), bw, 5, facecolor=XANH[6],
                           edgecolor="none", zorder=5))
    ax.add_patch(Rectangle((bx + bw * 0.43, by + bh * 0.34), bw * 0.14, 13,
                           facecolor=VANG, edgecolor="none", zorder=6))
    cx, cy, r = W * 0.73, 54, 13
    ax.add_patch(Circle((cx, cy), r, facecolor="#ffffff", alpha=0.94,
                        edgecolor=CAM, linewidth=3.2, zorder=7))
    ax.plot([cx + r * 0.72, cx + r * 1.7], [cy - r * 0.72, cy - r * 1.7],
            color=CAM, linewidth=5.0, solid_capstyle="round", zorder=7)


def ve_nha_tuyen_dung(ax, W):
    """Toà nhà văn phòng kèm hai nhân viên."""
    ax.add_patch(Rectangle((0, 0), W, 14, facecolor="#c3cedd",
                           edgecolor="none", zorder=1))
    toa_nha(ax, W * 0.14, 14, W * 0.40, 62, XANH[4], 3, 6)
    toa_nha(ax, W * 0.58, 14, W * 0.26, 44, XANH[2], 2, 4)
    for dx, mau in [(0.08, CAM), (0.93, LUC)]:
        x = W * dx
        ax.add_patch(Circle((x, 28), 5.2, facecolor=mau, edgecolor="none",
                            zorder=8))
        ax.add_patch(FancyBboxPatch((x - 5.4, 14), 10.8, 12,
                                    boxstyle="round,pad=0,rounding_size=3",
                                    facecolor=mau, edgecolor="none", zorder=8))


# ═══════════════════════════════════════════════ Chạy
print("Topic 1:")
hero_nha()
the(T1 / "vai_nguoi_ban.png", ve_nguoi_ban)
the(T1 / "vai_nguoi_mua.png", ve_nguoi_mua)
print("Project 2:")
hero_vanphong()
the(P2 / "vai_ung_vien.png", ve_ung_vien)
the(P2 / "vai_nha_tuyen_dung.png", ve_nha_tuyen_dung)
