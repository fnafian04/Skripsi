"""Pembagian data, pelatihan MNB, prediksi, dan evaluasi Nafinder.

Urutan kerja fungsi utama:
1. menghapus label duplikat;
2. membagi data berdasarkan ``review_id`` agar satu review tidak masuk ke
   data latih dan data uji sekaligus;
3. membuang teks preprocessing yang identik dengan data latih;
4. membentuk fitur TF-IDF dari data latih dan menerapkannya ke data uji;
5. melakukan oversampling hanya pada data latih;
6. melatih Multinomial Naive Bayes manual;
7. memprediksi data uji dan menghitung metrik evaluasi.

Urutan ini penting karena transformasi dan penyeimbangan data tidak boleh
memakai informasi dari data uji.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from .evaluation import evaluate
from .naive_bayes import MultinomialNaiveBayesManual, RandomOverSamplerManual, TfidfManual
from .split import group_shuffle_split


def fit_evaluate_manual(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    alpha: float = 0.1,
    min_df: int = 2,
    ros_target_ratio: float = 0.0,
    ros_class_ratio: dict[str, float] | None = None,
    progress: Callable | None = None,
) -> tuple[dict, MultinomialNaiveBayesManual, TfidfManual, tuple]:
    """Latih dan evaluasi MNB manual pada data hasil preprocessing.

    Konfigurasi utama menggunakan konfigurasi notebook:
    - alpha = 0.1;
    - min_df = 2;
    - Random Oversampling tidak digunakan.

    Model web memakai dua kelas: positif dan negatif.
    """
    # Validasi awal mencegah proses berjalan dengan hasil preprocessing yang
    # belum lengkap atau nama kolom yang salah.
    required = {"review_id", "teks_preprocessed", "sentimen"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom untuk klasifikasi tidak ada: {', '.join(sorted(missing))}")

    dedup_cols = (
        ["review_id", "teks_preprocessed", "aspek", "sentimen"]
        if "aspek" in df.columns
        else ["review_id", "teks_preprocessed", "sentimen"]
    )
    # Satu kombinasi review-teks-aspek-sentimen cukup dihitung sekali.
    data = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    stage_total = 5
    if progress:
        progress("Pembagian data", 0, stage_total,
                 f"{len(data):,} label setelah deduplikasi")

    # Group split menjaga semua bagian dari review yang sama berada di
    # kelompok yang sama sehingga evaluasi tidak terlalu optimistis.
    train_idx, test_idx = group_shuffle_split(
        data.review_id.to_numpy(), test_size=test_size, random_state=random_state
    )
    train = data.iloc[train_idx].reset_index(drop=True)
    test = data.iloc[test_idx].reset_index(drop=True)

    # Teks identik dikeluarkan dari testing untuk mengurangi kebocoran fitur.
    train_texts = set(train.teks_preprocessed)
    before = len(test)
    test = test[~test.teks_preprocessed.isin(train_texts)].reset_index(drop=True)
    removed = before - len(test)

    if test.empty:
        raise ValueError("Testing kosong setelah menghapus teks identik; periksa dataset.")
    if set(train.review_id) & set(test.review_id):
        raise AssertionError("Kebocoran review_id pada split.")
    if set(train.teks_preprocessed) & set(test.teks_preprocessed):
        raise AssertionError("Kebocoran teks_preprocessed pada split.")
    if set(train.sentimen) != set(test.sentimen):
        raise ValueError("Data latih dan uji harus sama-sama mengandung kelas positif dan negatif.")
    if progress:
        progress("Pembagian data selesai", 1, stage_total,
                 f"latih {len(train):,}, uji {len(test):,}")

    # Vectorizer hanya di-fit pada training; data testing hanya ditransformasi
    # memakai vocabulary dan IDF yang sudah dipelajari dari training.
    vec = TfidfManual(min_df=min_df)
    X_train = vec.fit_transform(train.teks_preprocessed.astype(str).tolist())
    X_test = vec.transform(test.teks_preprocessed.astype(str).tolist())
    if progress:
        progress("TF-IDF selesai", 2, stage_total,
                 f"{len(vec.vocabulary):,} kosakata")

    # Oversampling dilakukan setelah split dan hanya untuk memperkaya training.
    # Data testing harus tetap mencerminkan distribusi data asli.
    X_ros, y_ros = RandomOverSamplerManual(
        random_state,
        target_ratio=ros_target_ratio,
        class_ratio=ros_class_ratio or {},
    ).fit_resample(X_train, train.sentimen.to_numpy())
    if progress:
        progress("Oversampling selesai", 3, stage_total,
                 f"{len(y_ros):,} label latih")

    model = MultinomialNaiveBayesManual(alpha=alpha).fit(X_ros, y_ros)
    if progress:
        progress("Pelatihan MNB selesai", 4, stage_total,
                 f"{len(model.classes_)} kelas")

    # Model tidak pernah melihat label data testing selama training.
    prediction = model.predict(X_test)
    metrics = evaluate(test.sentimen.to_numpy(), prediction, labels=model.classes_)
    if progress:
        progress("Evaluasi selesai", 5, stage_total,
                 f"{len(prediction):,} prediksi")

    metrics.update({
        "split_unit": "review_id",
        "random_state": random_state,
        "jumlah_review_training": int(train.review_id.nunique()),
        "jumlah_review_testing": int(test.review_id.nunique()),
        "jumlah_label_training": len(train),
        "jumlah_label_testing": len(test),
        "baris_testing_dikeluarkan_teks_identik": removed,
        "jumlah_training_ros": len(y_ros),
        "kosakata_tfidf": len(vec.vocabulary),
        "alpha": alpha,
        "min_df": min_df,
        "ros_target_ratio": ros_target_ratio,
        "ros_class_ratio": ros_class_ratio or {},
    })
    return metrics, model, vec, (train, test, prediction)
