"""Orkestrator Nafinder: panggil preprocessing lalu klasifikasi."""
from __future__ import annotations

from typing import Callable

import pandas as pd

from .classification import fit_evaluate_manual
from .preprocessing import preprocess_and_label


def run_full_pipeline(csv_path: str, progress: Callable | None = None) -> dict:
    """Jalankan preprocessing dan klasifikasi dari satu file CSV."""
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    labels, prep_summary = preprocess_and_label(raw, progress=progress)
    metrics, model, vec, split = fit_evaluate_manual(labels, progress=progress)
    return {
        "labels": labels,
        "preprocessing": prep_summary,
        "metrics": metrics,
        "model": model,
        "vectorizer": vec,
        "split": split,
    }


# Re-export agar pemanggil lama tetap kompatibel.
__all__ = ["preprocess_and_label", "fit_evaluate_manual", "run_full_pipeline"]
