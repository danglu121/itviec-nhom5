"""Chuẩn bị ảnh thương hiệu ITViec cho ứng dụng.

Ba việc: tách nền trắng khỏi logo, cắt dải ảnh đầu trang, và cắt hai vùng
người từ ảnh minh hoạ để làm thẻ chọn vai.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

NGUON = Path("/home/dang.lu/Downloads/IT viec")
DICH = Path("/home/dang.lu/Personal/Final ML/GUI_Project2/figures")
DICH.mkdir(parents=True, exist_ok=True)

LOGO = NGUON / "itviec-white-01.jpg"
MINH_HOA = NGUON / "banner_1_5x-cc2334f39876362be5d9e5e0758c82de4fc8734a57d81dce6d633d6b93335ef7.jpg"
BIA = NGUON / "maxresdefault.jpg"


def bao(p: Path):
    im = Image.open(p)
    print(f"  {p.name:28s} {im.size[0]}x{im.size[1]}  "
          f"{p.stat().st_size / 1024:.0f} KB")


def cat_vien_trang(im: Image.Image, nguong: int = 246) -> Image.Image:
    """Cắt bỏ phần viền trắng thừa quanh ảnh."""
    a = np.asarray(im.convert("RGB"))
    khong_trang = (a < nguong).any(axis=2)
    hang = np.where(khong_trang.any(axis=1))[0]
    cot = np.where(khong_trang.any(axis=0))[0]
    if len(hang) == 0 or len(cot) == 0:
        return im
    return im.crop((cot[0], hang[0], cot[-1] + 1, hang[-1] + 1))


def bo_nen_trang(im: Image.Image) -> Image.Image:
    """Biến nền trắng thành trong suốt, giữ nguyên màu gốc của hình.

    Coi ảnh như hình đã được chồng lên nền trắng:
        quan_sat = alpha * mau + (1 - alpha) * 255
    Lấy alpha từ kênh tối nhất rồi giải ngược ra màu gốc. Cách này giữ được
    đỏ tươi của logo, trong khi phép trừ độ sáng thông thường sẽ làm nó xỉn.
    """
    a = np.asarray(im.convert("RGB")).astype(np.float64)
    toi_nhat = a.min(axis=2)
    alpha = 1.0 - toi_nhat / 255.0

    mau = np.zeros_like(a)
    co_hinh = alpha > 0.004
    for k in range(3):
        mau[..., k] = np.where(
            co_hinh,
            (a[..., k] - (1.0 - alpha) * 255.0) / np.maximum(alpha, 1e-6),
            255.0,
        )

    ra = np.dstack([np.clip(mau, 0, 255), np.clip(alpha * 255.0, 0, 255)])
    return Image.fromarray(ra.astype(np.uint8), mode="RGBA")


print("Ảnh gốc:")
for p in (LOGO, MINH_HOA, BIA):
    bao(p)

# ═══════════════════════════════════════════════════ 1. Logo
logo = cat_vien_trang(Image.open(LOGO))
logo = bo_nen_trang(logo)
logo.thumbnail((640, 640), Image.LANCZOS)
logo.save(DICH / "itviec_logo.png")
print(f"\n  logo       -> itviec_logo.png  {logo.size[0]}x{logo.size[1]}  "
      f"{(DICH / 'itviec_logo.png').stat().st_size / 1024:.0f} KB")

# ═════════════════════════════════════════ 2. Dải ảnh đầu trang
# Cắt phần trên của ảnh bìa: giữ nguyên cảnh văn phòng và logo góc trái,
# bỏ dòng khẩu hiệu chiến dịch ở nửa dưới.
bia = Image.open(BIA).convert("RGB")
hero = bia.crop((0, 0, bia.width, int(bia.height * 0.60)))
hero.save(DICH / "hero_itviec.jpg", quality=88, optimize=True)
print(f"  ảnh bìa    -> hero_itviec.jpg   {hero.size[0]}x{hero.size[1]}  "
      f"{(DICH / 'hero_itviec.jpg').stat().st_size / 1024:.0f} KB")

# ═══════════════════════════════════════════ 3. Hai thẻ chọn vai
mh = Image.open(MINH_HOA).convert("RGB")
W, H = mh.size
vung = {
    # Nhóm ba người quanh bàn đứng, dùng cho vai ứng viên
    "vai_ung_vien_anh.jpg": (0.13, 0.05, 0.57, 0.88),
    # Người cầm laptop và người phỏng vấn bên bảng, dùng cho nhà tuyển dụng
    "vai_nha_tuyen_dung_anh.jpg": (0.58, 0.10, 1.00, 0.98),
}
for ten, (x0, y0, x1, y1) in vung.items():
    c = mh.crop((int(W * x0), int(H * y0), int(W * x1), int(H * y1)))
    c.thumbnail((760, 760), Image.LANCZOS)
    c.save(DICH / ten, quality=88, optimize=True)
    print(f"  thẻ vai    -> {ten:26s} {c.size[0]}x{c.size[1]}  "
          f"{(DICH / ten).stat().st_size / 1024:.0f} KB")
