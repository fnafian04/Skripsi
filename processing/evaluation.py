"""Evaluasi klasifikasi manual: Confusion Matrix, Accuracy, Precision, Recall, dan F1-Score.
Implementasi dari dasar (from scratch) tanpa menggunakan library scikit-learn.

Formula Evaluasi (sesuai metodologi pada Bab III skripsi):

1. Confusion Matrix:
   Matrix N x N (N = jumlah kelas) di mana:
   - Baris  = label aktual (true label)
   - Kolom  = label prediksi (predicted label)
   - Diagonal (cm[i,i]) = True Positive (prediksi benar) untuk kelas i.

2. Precision per kelas c:
   Precision(c) = TP_c / (TP_c + FP_c)
                = jumlah prediksi benar kelas c / total prediksi kelas c
   Mengukur: dari semua yang diprediksi sebagai kelas c, berapa persen yang benar?

3. Recall per kelas c:
   Recall(c) = TP_c / (TP_c + FN_c)
             = jumlah prediksi benar kelas c / total data aktual kelas c
   Mengukur: dari semua yang sebenarnya kelas c, berapa persen yang berhasil dikenali?

4. F1-Score per kelas c (Harmonic Mean dari Precision dan Recall):
   F1(c) = 2 * Precision(c) * Recall(c) / (Precision(c) + Recall(c))
   Mengukur keseimbangan antara precision dan recall.

5. Accuracy keseluruhan:
   Accuracy = sum(TP_c for all c) / total_data
            = jumlah prediksi benar seluruh kelas / total data uji

6. Macro Average (tidak mempertimbangkan ketimpangan jumlah kelas):
   Macro-F1 = (1/N) * sum(F1(c) for all c)

7. Weighted Average (mempertimbangkan jumlah data setiap kelas):
   Weighted-F1 = sum(F1(c) * support(c) for all c) / total_data
   dengan support(c) = jumlah data aktual kelas c.
"""
from __future__ import annotations
import numpy as np


def evaluate(y_true, y_pred, labels=None) -> dict:
    """Menghitung seluruh metrik evaluasi klasifikasi secara manual.

    Args:
        y_true: Array label aktual (ground truth).
        y_pred: Array label hasil prediksi model.
        labels: Daftar nama kelas (urutan tetap: negatif, netral, positif).

    Returns:
        Dict berisi: confusion matrix, accuracy, per-class metrics,
        macro average, weighted average, dan jumlah data testing.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(labels or sorted(set(y_true) | set(y_pred)))
    n_classes = len(labels)

    # --- Membangun Confusion Matrix ---
    # cm[i, j] = jumlah data aktual kelas i yang diprediksi sebagai kelas j
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    index = {label: i for i, label in enumerate(labels)}
    for actual, predicted in zip(y_true, y_pred):
        cm[index[actual], index[predicted]] += 1

    total = len(y_true)

    # --- Precision, Recall, F1 per kelas ---
    per_class = {}
    for i, label in enumerate(labels):
        tp = int(cm[i, i])           # True Positive: diagonal confusion matrix
        support = int(cm[i].sum())   # Total data aktual kelas ini (TP + FN)
        predicted_count = int(cm[:, i].sum())  # Total prediksi kelas ini (TP + FP)

        precision = tp / predicted_count if predicted_count else 0.0
        recall    = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)

        per_class[label] = {
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "support":   support,
        }

    # --- Accuracy keseluruhan: jumlah diagonal / total ---
    accuracy = float(np.trace(cm) / total) if total else 0.0

    # --- Macro Average: rata-rata sederhana tanpa bobot jumlah kelas ---
    macro = {
        k: float(np.mean([r[k] for r in per_class.values()]))
        for k in ("precision", "recall", "f1")
    }

    # --- Weighted Average: rata-rata tertimbang jumlah data per kelas ---
    weighted = {
        k: float(sum(r[k] * r["support"] for r in per_class.values()) / total)
        if total else 0.0
        for k in ("precision", "recall", "f1")
    }

    return {
        "classes":         labels,
        "matrix":          cm.tolist(),
        "accuracy":        accuracy,
        "per_class":       per_class,
        "macro":           macro,
        "weighted":        weighted,
        "jumlah_testing":  total,
    }

