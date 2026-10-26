"""Bandingkan metode manual dengan scikit-learn sebagai audit opsional.

Tujuan: melihat perbedaan implementasi manual dan library pembanding pada
dataset serta konfigurasi yang sama. Hasilnya bukan target aplikasi dan tidak
menentukan hasil Colab maupun hasil runtime.

Penting:
- Script INI membutuhkan scikit-learn, tetapi HANYA untuk tujuan audit.
- Aplikasi Nafinder TIDAK membutuhkan sklearn (lihat requirements.txt).
- Path dataset diatur lewat variabel lingkungan DATASET_PREPROCESSED.

Cara menjalankan:
    pip install scikit-learn
    python processing/validate_methods.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.evaluation import evaluate  # noqa: E402
from processing.naive_bayes import (  # noqa: E402
    MultinomialNaiveBayesManual,
    RandomOverSamplerManual,
    TfidfManual,
)
from processing.split import group_shuffle_split  # noqa: E402

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import GroupShuffleSplit
except ImportError:  # pragma: no cover
    sys.exit(
        "Script audit ini butuh scikit-learn. Install dulu: pip install scikit-learn\n"
        "Aplikasi Nafinder sendiri TIDAK membutuhkan sklearn."
    )

PATH = os.getenv("DATASET_PREPROCESSED")
if not PATH:
    sys.exit("Set DATASET_PREPROCESSED ke path CSV hasil preprocessing terlebih dahulu.")

df = pd.read_csv(PATH, encoding="utf-8-sig")
df = df.drop_duplicates(
    subset=["review_id", "teks_preprocessed", "aspek", "sentimen"]
).reset_index(drop=True)
print(f"dataset : {PATH}")
print(f"baris   : {len(df):,}")

# --- Split grup: manual vs sklearn -----------------------------------------
a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
    df, groups=df.review_id))
tr_sk = df.iloc[a].reset_index(drop=True)
te_sk = df.iloc[b].reset_index(drop=True)
te_sk = te_sk[~te_sk.teks_preprocessed.isin(set(tr_sk.teks_preprocessed))].reset_index(drop=True)

tr_idx, te_idx = group_shuffle_split(df.review_id.to_numpy(), test_size=0.2, random_state=42)
tr_mn = df.iloc[tr_idx].reset_index(drop=True)
te_mn = df.iloc[te_idx].reset_index(drop=True)
te_mn = te_mn[~te_mn.teks_preprocessed.isin(set(tr_mn.teks_preprocessed))].reset_index(drop=True)

print()
print("=== SPLIT (berbasis review_id) ===")
print("manual  train/test:", len(tr_mn), len(te_mn),
      "| unique text:", len(set(tr_mn.teks_preprocessed)))
print("sklearn train/test:", len(tr_sk), len(te_sk),
      "| unique text:", len(set(tr_sk.teks_preprocessed)))
print("split sama:", tr_mn.equals(tr_sk) and te_mn.equals(te_sk))

# --- TF-IDF: manual vs sklearn ----------------------------------------------
manual = TfidfManual().fit_transform(tr_mn.teks_preprocessed.tolist())
vec = TfidfVectorizer(
    lowercase=True, sublinear_tf=True, smooth_idf=True, token_pattern=r"(?u)\b\w\w+\b"
)
ref = vec.fit_transform(tr_sk.teks_preprocessed)

print()
print("=== TF-IDF ===")
print("vocab manual / sklearn:", manual.shape, ref.shape)
print("kata bersama:", len(set(manual.vocabulary) & set(vec.vocabulary_)))
if manual.shape == ref.shape:
    print("selisih bobot maksimum:",
          float(np.abs(manual.toarray() - ref.toarray()).max()))
else:
    print("kosakata berbeda; selisih bobot tidak dibandingkan.")

# --- MNB manual + evaluasi pada split manual --------------------------------
print()
print("=== MNB manual ===")
X_ros, y_ros = RandomOverSamplerManual(42).fit_resample(
    manual, tr_mn.sentimen.to_numpy())
clf = MultinomialNaiveBayesManual(alpha=2).fit(X_ros, y_ros)

test_vec = TfidfManual()
test_vec.vocabulary = manual.vocabulary
test_vec.idf = manual.idf
X_test = test_vec.transform(te_mn.teks_preprocessed.tolist())
pred = clf.predict(X_test)

res = evaluate(te_mn.sentimen.to_numpy(), pred, labels=clf.classes_)
print("accuracy :", round(res["accuracy"], 6))
print("macro F1 :", round(res["macro"]["f1"], 6))
print("cm       :", res["matrix"])

print()
print("Perbandingan selesai. Selisih hasil dicatat untuk dianalisis, bukan dianggap error otomatis.")
