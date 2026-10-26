"""Pembagian data latih dan data uji berbasis grup (Group Shuffle Split).
Implementasi manual murni tanpa library scikit-learn.

Metodologi:
Dalam satu ulasan kafe, teks ulasan sering dipecah menjadi beberapa sub-kalimat.
Jika data diacak secara acak per sub-kalimat (random split biasa), sub-kalimat
dari ulasan yang sama bisa masuk ke data latih sekaligus data uji (data leakage).
Oleh karena itu, pembagian dilakukan berdasarkan `review_id` (grup):
seluruh sub-kalimat dari ulasan yang sama dijamin berada pada kelompok yang sama
(hanya di data latih ATAU hanya di data uji).
"""
from __future__ import annotations
import math
import numpy as np


def group_shuffle_split(groups, test_size: float = 0.2, random_state: int = 42):
    """Membagi data berdasarkan grup id (review_id) ke train dan test.

    Args:
        groups: Array atau list berisi identitas grup (misal: review_id) per baris data.
        test_size: Proporsi grup untuk data uji (default: 0.2 = 20%).
        random_state: Seed pengacakan untuk reprodusibilitas (default: 42).

    Returns:
        train_idx: Array indeks baris untuk data latih.
        test_idx: Array indeks baris untuk data uji.
    """
    groups_arr = np.asarray(groups)
    unique_groups, inverse = np.unique(groups_arr, return_inverse=True)
    n_groups = len(unique_groups)

    # Hitung jumlah grup yang dialokasikan untuk data uji (pembulatan ke atas)
    n_test_groups = int(math.ceil(test_size * n_groups))

    # Pengacakan grup unik dengan seed tetap
    rng = np.random.RandomState(random_state)
    permutation = rng.permutation(n_groups)

    test_groups = permutation[:n_test_groups]
    train_groups = permutation[n_test_groups:]

    # Petakan kembali grup ke indeks baris data asli
    train_idx = np.flatnonzero(np.isin(inverse, train_groups))
    test_idx = np.flatnonzero(np.isin(inverse, test_groups))

    return train_idx, test_idx

