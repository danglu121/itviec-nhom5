"""Đóng gói mô hình Project 2 thành artifact cho ứng dụng Streamlit.

Chạy một lần. Ứng dụng chỉ đọc kết quả, không huấn luyện lại lúc chạy.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Script này KHÔNG chạy khi triển khai. Nó chạy một lần ở máy phát triển,
# trên thư mục phân tích gốc có đủ dữ liệu và gensim, để sinh ra thư mục
# artifacts/. Repo triển khai chỉ cần kết quả, nên không kèm dữ liệu gốc.
SRC = Path(os.environ.get("ITVIEC_SRC", "../Project_2"))
OUT = Path(__file__).parent / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)
SEED, TOPK = 42, 20

NUMERIC = ["Rating", "Salary & benefits", "Training & learning",
           "Management cares about me", "Culture & fun", "Office & workspace"]


def mb(p: Path) -> float:
    return p.stat().st_size / 1048576


# ══════════════════════════════ YÊU CẦU 1 ══════════════════════════════
co = pd.read_parquet(SRC / "data" / "companies_clean.parquet")
co = co[co["text_clean"].str.strip().ne("")].reset_index(drop=True)
print(f"Kho công ty: {len(co)}")

# --- Láng giềng tính sẵn bằng phương pháp B (Gensim TF-IDF), mô hình đã chọn.
# Tính sẵn nên ứng dụng KHÔNG cần cài gensim lúc chạy.
from gensim import corpora, models, similarities  # noqa: E402

docs = [t.split() for t in co["text_clean"]]
dic = corpora.Dictionary(docs)
dic.filter_extremes(no_below=2, no_above=0.6)
bow = [dic.doc2bow(d) for d in docs]
tfidf = models.TfidfModel(bow)
index = similarities.SparseMatrixSimilarity(tfidf[bow], num_features=len(dic))
sim_b = np.vstack([index[tfidf[b]] for b in bow])
np.fill_diagonal(sim_b, -np.inf)

nn_idx = np.argsort(-sim_b, axis=1)[:, :TOPK].astype(np.int16)
nn_score = np.take_along_axis(sim_b, nn_idx.astype(np.int64), axis=1).astype(np.float32)
np.savez_compressed(OUT / "neighbors_gensim.npz", idx=nn_idx, score=nn_score)
print(f"  láng giềng Gensim: {nn_idx.shape}  "
      f"{mb(OUT / 'neighbors_gensim.npz'):.2f} MB")

# --- Vector hoá sklearn, phục vụ ô tìm kiếm bằng mô tả người dùng tự nhập.
# Phải dùng sklearn ở đây vì cần vector hoá văn bản MỚI lúc chạy; láng giềng
# tính sẵn ở trên không làm được việc đó.
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

vec = TfidfVectorizer(min_df=2, max_df=0.6, sublinear_tf=False)
X = vec.fit_transform(co["text_clean"])
joblib.dump({"vectorizer": vec, "matrix": X}, OUT / "search_tfidf.joblib", compress=3)
print(f"  tìm kiếm sklearn: {X.shape}  {mb(OUT / 'search_tfidf.joblib'):.2f} MB")

# --- Bảng hiển thị. Bỏ text_clean (chỉ máy đọc) và giữ mô tả gốc để hiện ra.
show = co[["id", "Company Name", "Company Type", "Company industry",
           "Company size", "Country", "Working days", "Overtime Policy",
           "Location", "Href", "text_raw", "lang"]].copy()
show["text_raw"] = show["text_raw"].str.slice(0, 1200)
show.to_parquet(OUT / "companies.parquet", index=False)
print(f"  bảng công ty: {show.shape}  {mb(OUT / 'companies.parquet'):.2f} MB")

# ══════════════════════════════ YÊU CẦU 2 ══════════════════════════════
rv = pd.read_parquet(SRC / "data" / "reviews_model.parquet")
tr, te = rv[rv.split == "train"], rv[rv.split == "test"]
ytr, yte = tr["label"].to_numpy(), te["label"].to_numpy()
print(f"\nReview: train {len(tr):,} | test {len(te):,}")

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
import xgboost as xgb  # noqa: E402

# Đúng cấu hình đã thắng ở notebook: XGBoost, văn bản + điểm số, không cân bằng.
pipe = Pipeline([
    ("pre", ColumnTransformer([
        ("text", TfidfVectorizer(min_df=3, max_df=0.7, ngram_range=(1, 2),
                                 max_features=20000, sublinear_tf=True), "text_clean"),
        ("num", StandardScaler(), NUMERIC),
    ], remainder="drop")),
    ("clf", xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.08, max_depth=6, subsample=0.85,
        colsample_bytree=0.7, reg_lambda=1.5, n_jobs=4, random_state=SEED,
        tree_method="hist", scale_pos_weight=1.0)),
])
t0 = time.perf_counter()
pipe.fit(tr, ytr)
print(f"  huấn luyện {time.perf_counter() - t0:.0f}s")

pred = pipe.predict(te)
prob = pipe.predict_proba(te)[:, 1]
tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
metrics = {
    "model": "XGBoost (văn bản + điểm số, không cân bằng lớp)",
    "n_train": int(len(tr)), "n_test": int(len(te)),
    "yes_pct": round(float(rv.label.mean() * 100), 1),
    "Accuracy": round(float(accuracy_score(yte, pred)), 4),
    "Precision_No": round(float(precision_score(yte, pred, pos_label=0, zero_division=0)), 4),
    "Recall_No": round(float(recall_score(yte, pred, pos_label=0, zero_division=0)), 4),
    "F1_No": round(float(f1_score(yte, pred, pos_label=0, zero_division=0)), 4),
    "ROC_AUC": round(float(roc_auc_score(yte, prob)), 4),
    "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    "baseline_accuracy": round(float(yte.mean()), 4),
}
joblib.dump(pipe, OUT / "recommend_clf.joblib", compress=3)
print(f"  bộ phân loại: {mb(OUT / 'recommend_clf.joblib'):.2f} MB")
print(f"  F1_No {metrics['F1_No']} | Recall_No {metrics['Recall_No']} | "
      f"Acc {metrics['Accuracy']} | AUC {metrics['ROC_AUC']}")

# ══════════════════════════════ SIÊU DỮ LIỆU ══════════════════════════════
yc1 = json.loads((SRC / "outputs" / "yc1_report.json").read_text(encoding="utf-8"))
meta = {
    "n_companies": int(len(co)),
    "topk": TOPK,
    "yeucau1": {
        "method": "B · TF-IDF (Gensim) + SparseMatrixSimilarity",
        "precision": yc1["precision"], "baseline": yc1["baseline"],
    },
    "yeucau2": metrics,
}
(OUT / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                   encoding="utf-8")

total = sum(mb(p) for p in OUT.iterdir())
print(f"\nTổng artifact: {total:.2f} MB trong {len(list(OUT.iterdir()))} tập tin")
