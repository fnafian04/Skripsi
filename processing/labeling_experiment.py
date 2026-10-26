"""Eksperimen labeling independen dari label pipeline yang tersimpan."""
from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np
import pandas as pd

from .text_processing import (
    KAMUS_NORMALISASI,
    KATA_ASPEK,
    NEGASI,
    NEGATIF,
    NEGATIF_FRASE,
    POSITIF,
    POSITIF_FRASE,
    TOLAK,
    TextPreprocessor,
    potong_template,
)


_SPLIT = re.compile(
    r"(?<!\d)[.!?]+(?!\d)|\n+|,\s*|"
    r"\s+(?:tapi|namun|tetapi|padahal|meski|walaupun|sedangkan)\s+",
    re.I,
)
_WORD = re.compile(r"[a-zA-Z']+")
_ASPECT_WORDS = {
    aspect: tuple(sorted((word.strip().lower() for word in words), key=len, reverse=True))
    for aspect, words in KATA_ASPEK.items()
}
_SERVICE_HINTS = ("diantar", "dilayani", "menunggu", "nunggu", "barista",
                  "kasir", "pegawai", "karyawan", "pelayan", "service")
_FOOD_HINTS = ("makanan", "minuman", "menu", "kopi", "rasa", "latte",
               "teh", "roti", "kue", "dessert", "nasi", "mie")


def _contains(text: str, phrase: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text))


def _aspects(text: str) -> list[str]:
    lowered = text.lower()
    found = [
        aspect for aspect, words in _ASPECT_WORDS.items()
        if any(_contains(lowered, word) for word in words)
    ]
    if any(_contains(lowered, word) for word in _SERVICE_HINTS):
        if "pelayanan" not in found:
            found.append("pelayanan")
        if "makanan_minuman" in found and not any(
            _contains(lowered, word) for word in _FOOD_HINTS
        ):
            found.remove("makanan_minuman")
    return found


def _split_aspect_clauses(text: str) -> list[str]:
    """Pisahkan koordinasi sederhana jika sisi kanan menyebut aspek lain."""
    parts = re.split(r"\s+(?:dan|serta|&)\s+", text, flags=re.I)
    if len(parts) == 1:
        return [text]
    result = []
    current = parts[0].strip()
    for part in parts[1:]:
        right = part.strip()
        if _aspects(right):
            if current:
                result.append(current)
            current = right
        else:
            current = f"{current} dan {right}".strip()
    if current:
        result.append(current)
    return result


def _sentiment(text: str, rating: Any) -> str:
    lowered = text.lower()
    score = sum(1 for phrase in POSITIF_FRASE if _contains(lowered, phrase))
    score -= sum(1 for phrase in NEGATIF_FRASE if _contains(lowered, phrase))
    tokens = _WORD.findall(lowered)
    for index, token in enumerate(tokens):
        if token in TOLAK:
            continue
        negated = any(item in tokens[max(0, index - 3):index] for item in NEGASI)
        if token in POSITIF:
            score += -1 if negated else 1
        elif token in NEGATIF:
            score += 1 if negated else -1
    if score > 0:
        return "positif"
    if score < 0:
        return "negatif"
    try:
        if rating is not None and float(rating) >= 5:
            return "positif"
        if rating is not None and float(rating) <= 1:
            return "negatif"
    except (TypeError, ValueError):
        pass
    return "netral"


def label_from_raw(
    raw_df: pd.DataFrame,
    progress: Callable | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Buat label eksperimen hanya dari CSV mentah."""
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
        for sentence in _SPLIT.split(review_text):
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
        raise ValueError("Eksperimen tidak menghasilkan label")
    summary = {
        "ulasan_awal": len(df),
        "ulasan_valid": len(valid),
        "label_setelah_preprocessing": len(labeled),
        "jumlah_review_berlabel": int(labeled.review_id.nunique()),
        "jumlah_kafe": int(labeled.placeId.nunique()),
        "distribusi_sentimen": labeled.sentimen.value_counts().to_dict(),
    }
    return labeled, summary
