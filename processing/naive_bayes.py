"""Implementasi metode manual TF-IDF, Random Oversampling, dan Multinomial Naive Bayes.
Ditulis secara manual dari dasar (from scratch) tanpa menggunakan library machine learning
(seperti scikit-learn, imbalanced-learn, scipy.sparse).

Formula dan Konfigurasi (sesuai klasifikasi_mnb.ipynb):
1. TF-IDF Manual:
   - Tokenisasi: pola regex r"(?u)\\b\\w\\w+\\b" (kata minimal 2 huruf, lowercase).
   - Term Frequency (Sublinear TF):
       tf(t, d) = 1 + ln(count(t, d))  jika count(t, d) > 0, selain itu 0.
   - Inverse Document Frequency (Smooth IDF):
       idf(t) = ln((1 + N) / (1 + df(t))) + 1.0
       dengan N = total dokumen, df(t) = frekuensi dokumen yang memuat kata t.
   - Pembobotan TF-IDF:
       tfidf(t, d) = tf(t, d) * idf(t)
   - Normalisasi L2 (Euclidean Norm):
       v_norm = v / sqrt(sum(v_i^2))

2. Random Oversampling Manual:
   - Diterapkan HANYA pada data latih (training data) untuk mencegah data leakage.
   - Menyeimbangkan jumlah sampel kelas minoritas ('negatif' dan 'netral') agar sama
     dengan jumlah sampel kelas mayoritas ('positif') melalui duplikasi acak dengan seed tetap.

3. Multinomial Naive Bayes Manual:
   - Log Class Prior:
       log P(c) = ln(N_c / N)
   - Log Likelihood dengan Laplace Smoothing (alpha = 1.0 pada baseline):
       log P(w|c) = ln((count(w, c) + alpha) / (sum_w count(w, c) + alpha * |V|))
       dengan |V| = total ukuran kosakata (fitur).
   - Prediksi Kelas (Maximum A Posteriori):
       y_pred = argmax_c [ log P(c) + sum_{w in d} tfidf(w, d) * log P(w|c) ]
"""
from __future__ import annotations
import math
import re
from collections import Counter
from typing import Sequence

import numpy as np
from .sparse_matrix import CsrMatrix


class TfidfManual:
    """Ekstraksi fitur teks menggunakan TF-IDF manual.

    Parameter:
        min_df: jumlah minimum dokumen yang harus memuat sebuah token agar
            token masuk vocabulary. Nilai 1 berarti semua token dipakai.
    """

    def __init__(self, min_df: int = 1):
        self.min_df = min_df
        self.vocabulary: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Memecah teks menjadi token kata minimal 2 karakter (lowercase)."""
        return re.findall(r"(?u)\b\w\w+\b", str(text).lower())

    def fit_transform(self, documents: Sequence[str]) -> CsrMatrix:
        """Membangun kamus kosakata (vocabulary), bobot IDF, dan matriks CSR data latih."""
        doc_freq: Counter[str] = Counter()
        tokenized = []
        for text in documents:
            tokens = self.tokenize(text)
            tokenized.append(tokens)
            doc_freq.update(set(tokens))

        # Filter kosakata berdasarkan batas minimum frekuensi dokumen (min_df)
        terms = sorted(term for term, count in doc_freq.items() if count >= self.min_df)
        self.vocabulary = {term: i for i, term in enumerate(terms)}
        n_docs = len(tokenized)

        # Smooth IDF: ln((1 + N) / (1 + df)) + 1
        self.idf = np.array([
            math.log((1 + n_docs) / (1 + doc_freq[t])) + 1.0
            for t in terms
        ], dtype=np.float64)

        return self._transform_tokens(tokenized)

    def transform(self, documents: Sequence[str]) -> CsrMatrix:
        """Mengubah dokumen baru (data uji) menjadi matriks CSR menggunakan kosakata & IDF yang sudah di-fit."""
        if self.idf is None:
            raise RuntimeError("TfidfManual harus di-fit dengan data training terlebih dahulu.")
        return self._transform_tokens([self.tokenize(x) for x in documents])

    def _transform_tokens(self, docs: list[list[str]]) -> CsrMatrix:
        """Menghitung sublinear TF, mengalikan dengan IDF, normalisasi L2, dan menyimpan ke CSR sparse."""
        data, indices, indptr = [], [], [0]
        for tokens in docs:
            counts = Counter(t for t in tokens if t in self.vocabulary)
            row = []
            for term, count in counts.items():
                j = self.vocabulary[term]
                # Sublinear TF: (1 + ln(count)) * idf
                value = (1.0 + math.log(count)) * self.idf[j]
                row.append((j, value))

            # Normalisasi L2 per dokumen
            norm = math.sqrt(sum(value * value for _, value in row))
            if norm:
                row = [(j, value / norm) for j, value in row]

            row.sort()
            indices.extend(j for j, _ in row)
            data.extend(value for _, value in row)
            indptr.append(len(data))

        return CsrMatrix(
            np.asarray(data, dtype=np.float64),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
            (len(docs), len(self.vocabulary))
        )


class RandomOverSamplerManual:
    """Penyeimbangan kelas dengan duplikasi acak pada data latih saja.

    Parameter:
        random_state: seed agar indeks yang diduplikasi selalu reproducible.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.sample_indices_: np.ndarray | None = None

    def fit_resample(self, X: CsrMatrix, y: Sequence[str]) -> tuple[CsrMatrix, np.ndarray]:
        """Menduplikasi secara acak sampel kelas minoritas hingga menyamai kelas mayoritas."""
        y_arr = np.asarray(y)
        rng = np.random.RandomState(self.random_state)
        classes, counts = np.unique(y_arr, return_counts=True)
        target = int(counts.max())

        indices = np.arange(len(y_arr), dtype=np.int64)
        extras = []
        for cls, count in zip(classes, counts):
            if count < target:
                class_indices = np.flatnonzero(y_arr == cls)
                extras.extend(rng.choice(class_indices, size=target - int(count), replace=True))

        self.sample_indices_ = np.concatenate((indices, np.asarray(extras, dtype=np.int64)))
        return X.take_rows(self.sample_indices_), y_arr[self.sample_indices_]


class MultinomialNaiveBayesManual:
    """MNB manual dengan Laplace smoothing.

    Parameter:
        alpha: konstanta smoothing yang ditambahkan ke setiap hitungan fitur.
            Alpha 1.0 digunakan sebagai konfigurasi baseline klasifikasi_mnb.ipynb.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.classes_: list[str] = []
        self.log_prior_: np.ndarray | None = None
        self.log_likelihood_: np.ndarray | None = None

    def fit(self, X: CsrMatrix, y: Sequence[str]):
        """Melatih model: menghitung log prior per kelas dan log likelihood per kata."""
        y_arr = np.asarray(y)
        self.classes_ = sorted(set(y_arr))
        n_docs, n_features = X.shape

        self.log_prior_ = np.empty(len(self.classes_), dtype=np.float64)
        self.log_likelihood_ = np.empty((len(self.classes_), n_features), dtype=np.float64)

        for ci, cls in enumerate(self.classes_):
            mask = (y_arr == cls)
            # Log Prior: ln(N_c / N)
            self.log_prior_[ci] = math.log(mask.sum() / n_docs)
            # Total bobot tiap kata pada kelas cls
            counts = X.column_sums(mask)
            # Laplace smoothing: (count + alpha) / (total_tokens + alpha * n_features)
            denom = counts.sum() + self.alpha * n_features
            self.log_likelihood_[ci] = np.log((counts + self.alpha) / denom)

        return self

    def predict(self, X: CsrMatrix, batch_size: int = 4096) -> np.ndarray:
        """Memprediksi label sentimen untuk data uji menggunakan Maximum A Posteriori (MAP)."""
        if self.log_prior_ is None or self.log_likelihood_ is None:
            raise RuntimeError("Model Multinomial Naive Bayes belum di-fit.")

        preds = []
        for start in range(0, X.shape[0], batch_size):
            # Perkalian dot: X @ log_likelihood.T
            scores = X[start:start + batch_size].dot_dense(self.log_likelihood_.T)
            # Tambahkan log prior
            scores += self.log_prior_
            # Ambil indeks kelas dengan skor probabilitas tertinggi (argmax)
            preds.extend(self.classes_[i] for i in scores.argmax(axis=1))

        return np.asarray(preds)
