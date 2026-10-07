"""Tiền xử lý văn bản nhận diện ngôn ngữ.

Giống hệt lớp tiền xử lý dùng lúc huấn luyện trong notebook. Bắt buộc phải
giống, vì bộ phân loại đã học trên văn bản được làm sạch theo đúng cách này;
làm sạch khác đi lúc dự đoán sẽ cho kết quả sai lệch.
"""

from __future__ import annotations

import re
import unicodedata
import warnings

# pyvi không kèm danh sách từ dừng tiếng Việt nên phải tự định nghĩa.
# CỐ Ý GIỮ LẠI các từ phủ định ("không", "chưa", "chẳng") vì chúng đảo ngược
# hoàn toàn nghĩa của một câu đánh giá.
VI_STOPWORDS = {
    "và", "là", "của", "có", "được", "các", "những", "một", "cho", "với",
    "trong", "đã", "khi", "này", "đó", "cũng", "nên", "thì", "ở", "về",
    "để", "từ", "rất", "nhiều", "lại", "vào", "ra", "nhưng", "mà", "hay",
    "còn", "sẽ", "bị", "bởi", "tại", "theo", "như", "nếu", "vì", "do",
    "đang", "vẫn", "mình", "tôi", "em", "anh", "chị", "bạn", "họ", "ai",
    "gì", "nào", "sao", "thế", "vậy", "ấy", "kia", "đây", "đấy", "ạ",
    "nhé", "nha", "à", "ừ", "ok", "oke", "thôi", "chỉ", "luôn", "hơi",
    "khá", "quá", "lắm", "cực", "siêu", "hẳn", "toàn", "cả", "mọi",
    "mỗi", "từng", "vài", "dăm", "chút", "ít", "nữa", "thêm", "bớt",
}
EN_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for",
    "with", "about", "into", "through", "to", "from", "in", "on", "off",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does",
    "did", "have", "has", "had", "will", "would", "can", "could", "shall",
    "should", "may", "might", "must", "this", "that", "these", "those",
    "it", "its", "we", "our", "you", "your", "they", "their", "he", "she",
    "as", "so", "than", "then", "there", "here", "when", "where", "which",
    "who", "whom", "what", "how", "all", "any", "both", "each", "more",
    "most", "other", "some", "such", "very", "too", "also", "just", "only",
}
# Giữ lại "no", "not", "never" vì cùng lý do phủ định.

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_NUM = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_NONWORD = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")

_tok = None


def viet_score(text: str) -> float:
    """Tỷ lệ ký tự mang dấu thanh hoặc dấu mũ trên tổng số ký tự.

    Tiếng Việt viết có dấu cho tỷ lệ cao, tiếng Anh cho gần 0. Phân bố của
    chỉ số này trên dữ liệu huấn luyện là lưỡng cực rõ rệt, với khoảng trũng
    quanh 0,04, nên ngưỡng đặt ở đó ít nhạy cảm với lựa chọn.
    """
    text = str(text)
    if not text:
        return 0.0
    marks = sum(1 for c in unicodedata.normalize("NFD", text)
                if unicodedata.category(c) == "Mn")
    return marks / len(text)


def detect_lang(text: str, threshold: float = 0.04) -> str:
    return "vi" if viet_score(text) >= threshold else "en"


def basic_clean(text: str) -> str:
    """Bước làm sạch chung, chạy trước khi rẽ nhánh theo ngôn ngữ."""
    text = str(text).lower()
    for pat in (_URL, _EMAIL, _NUM, _NONWORD):
        text = pat.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def vi_tokenize(text: str) -> list[str]:
    """Tách từ tiếng Việt bằng pyvi, ví dụ 'môi trường' thành 'môi_trường'.

    Tiếng Việt có nhiều từ ghép hai âm tiết; tách theo khoảng trắng sẽ làm
    vỡ nghĩa. Nạp pyvi theo kiểu lười để lần khởi động đầu của ứng dụng
    không phải chờ.
    """
    global _tok
    if _tok is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pyvi import ViTokenizer
        _tok = ViTokenizer
    return _tok.tokenize(text).split()


def preprocess(text: str, min_len: int = 2) -> tuple[str, str]:
    """Làm sạch một văn bản. Trả về (chuỗi token, ngôn ngữ đã nhận diện)."""
    raw = str(text) if text is not None else ""
    lang = detect_lang(raw)
    cleaned = basic_clean(raw)
    if not cleaned:
        return "", lang
    if lang == "vi":
        tokens, stop = vi_tokenize(cleaned), VI_STOPWORDS
    else:
        tokens, stop = cleaned.split(), EN_STOPWORDS
    tokens = [t for t in tokens
              if t not in stop and len(t.replace("_", "")) >= min_len]
    return " ".join(tokens), lang
