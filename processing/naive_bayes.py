"""Implementasi manual TF-IDF, Random Oversampling, dan Multinomial Naive Bayes.

Modul ini mengubah teks menjadi angka dengan TF-IDF, memperbaiki ketimpangan
jumlah label pada data latih dengan duplikasi acak terkontrol, lalu melatih
Multinomial Naive Bayes untuk memilih sentimen dengan skor probabilitas terbesar.
Seluruh perhitungan inti ditulis dari dasar tanpa scikit-learn, imbalanced-learn,
atau scipy.sparse, sehingga setiap tahap metode dapat dijelaskan dan ditelusuri
langsung dari kode.

Formula dan Konfigurasi (sesuai klasifikasi_mnb.ipynb):
1. TF-IDF Manual:
   - Tokenisasi: unigram dan bigram dari kata minimal 2 huruf.
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
   - Hanya digunakan bila konfigurasi eksperimen memintanya.
   - Konfigurasi utama web mengikuti notebook tanpa oversampling.

3. Multinomial Naive Bayes Manual:
   - Log Class Prior:
       log P(c) = ln(N_c / N)
   - Log Likelihood dengan Laplace Smoothing (alpha = 0.1 sesuai model utama):
       log P(w|c) = ln((count(w, c) + alpha) / (sum_w count(w, c) + alpha * |V|))
       dengan |V| = total ukuran kosakata (fitur).
   - Prediksi Kelas (Maximum A Posteriori):
       y_pred = argmax_c [ log P(c) + sum_{w in d} tfidf(w, d) * log P(w|c) ]
"""
from __future__ import annotations

# Fungsi matematika untuk logaritma dan akar kuadrat.
import math
# Regex untuk tokenisasi kata.
import re
# Counter menghitung frekuensi kata secara efisien.
from collections import Counter
# Sequence menerima list, tuple, atau struktur berurutan lain.
from typing import Sequence

# NumPy dipakai untuk array numerik dan operasi matriks.
import numpy as np
# CSR menyimpan matriks sparse agar fitur yang bernilai nol tidak boros memori.
from .sparse_matrix import CsrMatrix


class TfidfManual:
    """Ekstraksi fitur teks menggunakan TF-IDF manual.

    Parameter:
        min_df: jumlah minimum dokumen yang harus memuat sebuah token agar
            token masuk vocabulary. Nilai 1 berarti semua token dipakai.
    """

    def __init__(self, min_df: int = 1):
        self.min_df = min_df  # Batas minimum kemunculan dokumen untuk sebuah kata.
        self.vocabulary: dict[str, int] = {}  # Peta kata ke indeks fitur.
        self.idf: np.ndarray | None = None  # Bobot IDF setelah data latih dipelajari.

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Memecah teks menjadi unigram dan bigram seperti konfigurasi notebook."""
        unigrams = re.findall(r"(?u)\b\w\w+\b", str(text).lower())
        bigrams = [
            f"{unigrams[i]} {unigrams[i + 1]}"
            for i in range(len(unigrams) - 1)
        ]
        return unigrams + bigrams

    def fit_transform(self, documents: Sequence[str]) -> CsrMatrix:
        """Membangun kamus kosakata (vocabulary), bobot IDF, dan matriks CSR data latih."""
        doc_freq: Counter[str] = Counter()  # Jumlah dokumen yang memuat setiap kata.
        tokenized = []  # Menyimpan token tiap dokumen untuk tahap berikutnya.
        for text in documents:
            tokens = self.tokenize(text)  # Ubah satu dokumen menjadi daftar kata.
            tokenized.append(tokens)  # Simpan token agar tidak menghitung ulang.
            doc_freq.update(set(tokens))  # Set mencegah satu dokumen dihitung dua kali.

        # Filter kosakata berdasarkan batas minimum frekuensi dokumen (min_df)
        terms = sorted(term for term, count in doc_freq.items() if count >= self.min_df)
        self.vocabulary = {term: i for i, term in enumerate(terms)}  # Kata menjadi indeks fitur.
        n_docs = len(tokenized)  # Total dokumen pada data latih.

        # Smooth IDF: ln((1 + N) / (1 + df)) + 1
        self.idf = np.array([  # Kata yang jarang muncul mendapat bobot lebih besar.
            math.log((1 + n_docs) / (1 + doc_freq[t])) + 1.0
            for t in terms
        ], dtype=np.float64)

        return self._transform_tokens(tokenized)  # Bentuk matriks TF-IDF data latih.

    def transform(self, documents: Sequence[str]) -> CsrMatrix:
        """Mengubah dokumen baru (data uji) menjadi matriks CSR menggunakan kosakata & IDF yang sudah di-fit."""
        if self.idf is None:
            raise RuntimeError("TfidfManual harus di-fit dengan data training terlebih dahulu.")
        # Data baru memakai vocabulary dan IDF data latih agar tidak terjadi leakage.
        return self._transform_tokens([self.tokenize(x) for x in documents])

    def _transform_tokens(self, docs: list[list[str]]) -> CsrMatrix:
        """Menghitung sublinear TF, mengalikan dengan IDF, normalisasi L2, dan menyimpan ke CSR sparse."""
        data, indices, indptr = [], [], [0]  # Tiga komponen penyimpanan matriks CSR.
        for tokens in docs:
            counts = Counter(t for t in tokens if t in self.vocabulary)  # Frekuensi fitur.
            row = []  # Pasangan indeks fitur dan nilai TF-IDF satu dokumen.
            for term, count in counts.items():
                j = self.vocabulary[term]  # Ambil posisi kata dalam vocabulary.
                # Sublinear TF: (1 + ln(count)) * idf
                value = (1.0 + math.log(count)) * self.idf[j]
                row.append((j, value))  # Simpan bobot fitur yang tidak nol.

            # Normalisasi L2 per dokumen
            norm = math.sqrt(sum(value * value for _, value in row))
            if norm:
                row = [(j, value / norm) for j, value in row]  # Samakan skala panjang dokumen.

            row.sort()  # CSR membutuhkan indeks fitur yang terurut.
            indices.extend(j for j, _ in row)  # Simpan indeks kolom yang terisi.
            data.extend(value for _, value in row)  # Simpan nilai TF-IDF.
            indptr.append(len(data))  # Penanda akhir baris saat ini.

        return CsrMatrix(  # Bangun matriks sparse yang hemat memori.
            np.asarray(data, dtype=np.float64),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
            (len(docs), len(self.vocabulary))
        )


class RandomOverSamplerManual:
    """Penyeimbangan kelas dengan duplikasi acak pada data latih saja.

    Parameter:
        random_state: seed agar indeks yang diduplikasi selalu reproducible.
        target_ratio: fraksi jarak kelas minoritas ke mayoritas yang dicapai
            setelah resampling (1.0 = disamakan penuh dengan mayoritas,
            0.6 = hanya menutup 60% selisih, 0.0 = tanpa duplikasi).
    """

    def __init__(self, random_state: int = 42, target_ratio: float = 1.0,
                 class_ratio: dict[str, float] | None = None):
        self.random_state = random_state  # Seed membuat pemilihan sampel konsisten.
        self.target_ratio = float(target_ratio)  # Fraksi selisih yang akan ditutup.
        # Rasio per kelas menimpa target_ratio untuk kelas tertentu.
        # Contoh: {"negatif": 0.65} memakai 0,65 untuk kelas negatif saja.
        self.class_ratio = dict(class_ratio or {})  # Rasio khusus untuk kelas tertentu.
        self.sample_indices_: np.ndarray | None = None  # Indeks hasil resampling.

    def _ratio_for(self, cls: str) -> float:
        return float(self.class_ratio.get(cls, self.target_ratio))  # Ambil rasio kelas.

    def fit_resample(self, X: CsrMatrix, y: Sequence[str]) -> tuple[CsrMatrix, np.ndarray]:
        """Menduplikasi secara acak sampel kelas minoritas mengikuti target_ratio per kelas."""
        y_arr = np.asarray(y)  # Label menjadi array agar mudah dibandingkan.
        rng = np.random.RandomState(self.random_state)  # Generator acak reproducible.
        classes, counts = np.unique(y_arr, return_counts=True)  # Kelas dan jumlahnya.
        target_gap = float(counts.max())  # Jumlah kelas mayoritas sebagai acuan.

        indices = np.arange(len(y_arr), dtype=np.int64)  # Pertahankan semua sampel asli.
        extras = []  # Menyimpan indeks tambahan kelas minoritas.
        for cls, count in zip(classes, counts):
            if count < target_gap:
                ratio = self._ratio_for(str(cls))  # Ambil rasio penambahan kelas ini.
                target = int(round(count + (target_gap - count) * ratio))
                if target <= int(count):
                    continue
                class_indices = np.flatnonzero(y_arr == cls)  # Lokasi data kelas minoritas.
                extras.extend(rng.choice(class_indices, size=target - int(count), replace=True))

        self.sample_indices_ = np.concatenate((indices, np.asarray(extras, dtype=np.int64)))
        # Duplikasi hanya diterapkan ke X dan y training, bukan data testing.
        return X.take_rows(self.sample_indices_), y_arr[self.sample_indices_]


class MultinomialNaiveBayesManual:
    """MNB manual dengan Laplace smoothing.

    Parameter:
        alpha: konstanta smoothing yang ditambahkan ke setiap hitungan fitur.
        Konfigurasi model utama menggunakan alpha 0.1.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = float(alpha)  # Smoothing agar fitur yang belum muncul tidak bernilai nol.
        self.classes_: list[str] = []  # Daftar kelas sentimen yang dipelajari.
        self.log_prior_: np.ndarray | None = None  # Log peluang awal tiap kelas.
        self.log_likelihood_: np.ndarray | None = None  # Log peluang fitur tiap kelas.

    def fit(self, X: CsrMatrix, y: Sequence[str]):
        """Melatih model: menghitung log prior per kelas dan log likelihood per kata."""
        y_arr = np.asarray(y)  # Ubah label menjadi array untuk masking kelas.
        self.classes_ = sorted(set(y_arr))  # Urutkan label agar indeks stabil.
        n_docs, n_features = X.shape  # Ambil jumlah dokumen dan fitur.

        self.log_prior_ = np.empty(len(self.classes_), dtype=np.float64)
        self.log_likelihood_ = np.empty((len(self.classes_), n_features), dtype=np.float64)

        for ci, cls in enumerate(self.classes_):
            mask = (y_arr == cls)  # Tandai baris training yang termasuk kelas ini.
            # Log Prior: ln(N_c / N)
            self.log_prior_[ci] = math.log(mask.sum() / n_docs)
            # Total bobot tiap kata pada kelas cls
            counts = X.column_sums(mask)  # Jumlah bobot fitur pada kelas ini.
            # Laplace smoothing: (count + alpha) / (total_tokens + alpha * n_features)
            denom = counts.sum() + self.alpha * n_features
            self.log_likelihood_[ci] = np.log((counts + self.alpha) / denom)  # Simpan log likelihood.

        return self

    def predict(self, X: CsrMatrix, batch_size: int = 4096) -> np.ndarray:
        """Memprediksi label sentimen untuk data uji menggunakan Maximum A Posteriori (MAP)."""
        if self.log_prior_ is None or self.log_likelihood_ is None:
            raise RuntimeError("Model Multinomial Naive Bayes belum di-fit.")

        preds = []  # Menampung hasil prediksi seluruh batch.
        for start in range(0, X.shape[0], batch_size):
            # Perkalian dot: X @ log_likelihood.T
            scores = X[start:start + batch_size].dot_dense(self.log_likelihood_.T)
            # Tambahkan log prior
            scores += self.log_prior_  # Skor akhir = prior + likelihood.
            # Ambil indeks kelas dengan skor probabilitas tertinggi (argmax)
            preds.extend(self.classes_[i] for i in scores.argmax(axis=1))

        return np.asarray(preds)  # Kembalikan label prediksi untuk evaluasi.
