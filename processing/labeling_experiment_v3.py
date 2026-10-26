"""Eksperimen labeling v3 dengan pemisahan aspek yang lebih eksplisit."""
from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd

from .labeling_experiment import (
    _ASPECT_WORDS,
    _contains,
    _sentiment,
)
from .text_processing import TextPreprocessor, potong_template

_SENTENCE_SPLIT = re.compile(
    r"(?<!\d)[.!?]+(?!\d)|\n+|,\s*|"
    r"\s+(?:tapi|namun|tetapi|padahal|meski|walaupun|sedangkan)\s+",
    re.I,
)
_COORDINATING = re.compile(r"\s+(?:dan|serta|&)\s+", re.I)
_SERVICE_HINTS = (
    "diantar", "dilayani", "menunggu", "nunggu", "barista",
    "kasir", "pegawai", "karyawan", "pelayan", "service",
)
_FOOD_EVALUATION = (
    "enak", "lezat", "nikmat", "rasa", "pahit", "manis", "asin",
    "pedas", "gurih", "segar", "hambar", "dingin", "menu",
)


def _aspects(text: str) -> list[str]:
    lowered = text.lower()
    found = [
        aspect for aspect, words in _ASPECT_WORDS.items()
        if any(_contains(lowered, word) for word in words)
    ]
    service_context = any(_contains(lowered, word) for word in _SERVICE_HINTS)
    food_context = any(_contains(lowered, word) for word in _FOOD_EVALUATION)
    if service_context and not food_context and "makanan_minuman" in found:
        found.remove("makanan_minuman")
    if service_context and "pelayanan" not in found:
        found.append("pelayanan")
    return found


def _split_aspect_clauses(text: str) -> list[str]:
    parts = [part.strip() for part in _COORDINATING.split(text) if part.strip()]
    if len(parts) == 1:
        return parts
    clauses: list[str] = []
    current = parts[0]
    for part in parts[1:]:
        if _aspects(part):
            clauses.append(current)
            current = part
        else:
            current = f"{current} dan {part}"
    clauses.append(current)
    return clauses


def label_from_raw(
    raw_df: pd.DataFrame,
    progress: Callable | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Buat label v3 hanya dari data mentah."""
    required = {
        "placeId", "nama_kafe", "teks_ulasan", "rating_ulasan",
        "tanggal_ulasan", "kafe_id", "valid",
    }
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"Kolom CSV mentah tidak lengkap: {', '.join(sorted(missing))}")

    df = raw_df.copy().reset_index(drop=True)
    df["review_id"] = np.arange(len(df), dtype=np.int64)
    df["valid"] = df["valid"].astype(str).str.lower().eq("true")
    df["teks_ulasan"] = df["teks_ulasan"].fillna("").astype(str)
    valid = df[df["valid"]].copy().reset_index(drop=True)
    preprocessor = TextPreprocessor()
    output = []

    for position, row in enumerate(valid.itertuples(index=False), 1):
        review_text = potong_template(row.teks_ulasan)
        sentences = _SENTENCE_SPLIT.split(review_text)
        for sentence in sentences:
            sentence = sentence.strip(" ;:!?\"'“”‘’")
            if not sentence:
                continue
            for clause in _split_aspect_clauses(sentence):
                aspects = _aspects(clause)
                processed = preprocessor.preprocess(clause)
                if not aspects or not processed:
                    continue
                sentiment = _sentiment(clause, row.rating_ulasan)
                for aspect in aspects:
                    output.append({
                        "review_id": int(row.review_id),
                        "placeId": row.placeId,
                        "nama_kafe": row.nama_kafe,
                        "kafe_id": row.kafe_id,
                        "rating_ulasan": row.rating_ulasan,
                        "asal_sub_kalimat": review_text,
                        "sub_kalimat": clause,
                        "aspek": aspect,
                        "sentimen": sentiment,
                        "teks_preprocessed": processed,
                    })
        if progress and (position % 1000 == 0 or position == len(valid)):
            progress(position, len(valid), len(output))

    labeled = pd.DataFrame(output)
    if labeled.empty:
        raise ValueError("Eksperimen v3 tidak menghasilkan label")
    summary = {
        "ulasan_awal": len(df),
        "ulasan_valid": len(valid),
        "label_setelah_preprocessing": len(labeled),
        "jumlah_review_berlabel": int(labeled.review_id.nunique()),
        "jumlah_kafe": int(labeled.placeId.nunique()),
        "distribusi_aspek": labeled.aspek.value_counts().to_dict(),
        "distribusi_sentimen": labeled.sentimen.value_counts().to_dict(),
        "subkalimat_multi_aspek": int(
            (labeled.groupby(["review_id", "sub_kalimat"]).aspek.nunique() > 1).sum()
        ),
    }
    return labeled, summary
