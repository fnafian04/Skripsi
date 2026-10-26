"""Struktur data Compressed Sparse Row (CSR) minimal untuk representasi matriks TF-IDF.
Implementasi manual murni tanpa ketergantungan library scipy.sparse.

Urgensi metodologis:
Pada dataset ulasan berukuran besar (misalnya 240.000+ sub-kalimat dan 12.000+ kosakata),
representasi matriks padat (dense matrix) 240.000 x 12.000 angka floating-point (8 byte)
akan membutuhkan memori RAM sekitar:
    240.000 * 12.000 * 8 byte ≈ 23.040.000.000 byte ≈ 23 GB RAM.
Hal tersebut berisiko menyebabkan crash (Out Of Memory).

Dengan representasi CSR, hanya nilai bukan nol (non-zero) yang disimpan:
- `data`: array 1D berisi nilai bobot TF-IDF bukan nol.
- `indices`: array 1D berisi indeks kolom (fitur kata) untuk setiap nilai di `data`.
- `indptr`: array 1D penunjuk indeks awal dan akhir baris pada array `data`.
- `shape`: dimensi (jumlah_baris, jumlah_kolom).

Penggunaan RAM berkurang drastis dari ~23 GB menjadi hanya puluhan megabyte (~20 MB).
"""
from __future__ import annotations
import numpy as np


class CsrMatrix:
    """Representasi matriks jarang (Sparse Matrix) berformat CSR (Compressed Sparse Row)."""

    __slots__ = ('data', 'indices', 'indptr', 'shape')

    def __init__(self, data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, shape: tuple[int, int]):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = shape

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.take_rows(np.arange(self.shape[0])[key])
        if isinstance(key, (list, np.ndarray)):
            return self.take_rows(key)
        return self.take_rows([key])

    def row(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Mengambil pasangan (kolom, nilai) untuk baris ke-i."""
        start = self.indptr[i]
        end = self.indptr[i + 1]
        return self.indices[start:end], self.data[start:end]

    def take_rows(self, rows) -> CsrMatrix:
        """Mengambil subset baris tertentu (digunakan pada split data dan oversampling)."""
        data_list = []
        indices_list = []
        indptr_list = [0]
        row_indices = np.asarray(rows, dtype=np.int64)

        for i in row_indices:
            cols, vals = self.row(int(i))
            indices_list.append(cols)
            data_list.append(vals)
            indptr_list.append(indptr_list[-1] + len(cols))

        new_data = np.concatenate(data_list) if data_list else np.zeros(0, dtype=np.float64)
        new_indices = np.concatenate(indices_list) if indices_list else np.zeros(0, dtype=np.int64)
        new_indptr = np.asarray(indptr_list, dtype=np.int64)

        return CsrMatrix(new_data, new_indices, new_indptr, (len(row_indices), self.shape[1]))

    def column_sums(self, mask: np.ndarray) -> np.ndarray:
        """Menghitung jumlah nilai per kolom untuk baris-baris yang memenuhi mask tertentu.
        Digunakan dalam pelatihan Multinomial Naive Bayes untuk menghitung total bobot kata per kelas.
        """
        out = np.zeros(self.shape[1], dtype=np.float64)
        for i in np.flatnonzero(mask):
            cols, vals = self.row(int(i))
            np.add.at(out, cols, vals)
        return out

    def dot_dense(self, weights: np.ndarray) -> np.ndarray:
        """Perkalian matriks jarang dengan matriks padat: X_sparse @ weights_dense.
        Digunakan dalam prediksi Naive Bayes: menghitung dot product bobot fitur dengan log likelihood.
        """
        out = np.zeros((self.shape[0], weights.shape[1]), dtype=np.float64)
        for i in range(self.shape[0]):
            cols, vals = self.row(i)
            if len(cols):
                out[i] = vals @ weights[cols]
        return out

