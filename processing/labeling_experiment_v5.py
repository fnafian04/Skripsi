"""Eksperimen labeling v5 berdasarkan koreksi manual terhadap v4."""
from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd

from . import labeling_experiment_v4 as v4
from .text_processing import TextPreprocessor, potong_template

_WORD = re.compile(r"[a-zA-Z']+")
_INFORMATIONAL = (
    "ada ", "tersedia", "yaitu ", "dipesan", "pesan ",
    "pilihan ", "tersedia ",
)
_NEGATIVE_CONTEXT = (
    "wifi e troble", "wifi trouble", "wifi troble", "kurang cocok",
    "sejam lebih nunggu", "baru diinfokan", "tidak sepadan",
    "kurang endeus", "kurang enak", "sempit", "gelap",
    "tinggal dikit", "tinggal sedikit", "jadi ga jadi", "jadi tidak jadi",
    "lebih banyak colokan",
)
_POSITIVE_CONTEXT = ("terawat baik", "jinak lucu", "cocok buat", "cocok untuk")


def _aspects(text: str) -> list[str]:
    lowered = text.lower()
    found = v4._aspects(text)
    wfc = v4._has_any(lowered, v4._WFC)
    explicit_food_evaluation = v4._has_any(
        lowered, ("rasa", "enak", "lezat", "pahit", "manis", "asin", "hambar")
    )
    meeting_context = v4._has_any(lowered, ("meeting", "nugas", "kerja laptop", "laptopan"))
    if (wfc or meeting_context) and "fasilitas_wfc" not in found:
        found.append("fasilitas_wfc")
    if (wfc or meeting_context) and (
        not explicit_food_evaluation or "enak buat meeting" in lowered
    ):
        found = [item for item in found if item != "makanan_minuman"]
    if v4._has_any(lowered, ("lokasi", "satu lokasi", "indoor", "rumahan")):
        if "suasana" not in found:
            found.append("suasana")
        found = [item for item in found if item != "makanan_minuman"]
    return list(dict.fromkeys(found))


def _sentiment(text: str, rating: object) -> str:
    lowered = text.lower()
    if any(phrase in lowered for phrase in _NEGATIVE_CONTEXT):
        return "negatif"
    if any(phrase in lowered for phrase in _POSITIVE_CONTEXT):
        return "positif"
    if any(phrase in lowered for phrase in _INFORMATIONAL):
        if not v4._has_any(lowered, v4.POSITIF_FRASE | v4.NEGATIF_FRASE):
            return "netral"
    if "lebih banyak colokan" in lowered:
        return "negatif"
    score = sum(1 for phrase in v4.POSITIF_FRASE if v4._contains(lowered, phrase))
    score -= sum(1 for phrase in v4.NEGATIF_FRASE if v4._contains(lowered, phrase))
    tokens = _WORD.findall(lowered)
    for index, token in enumerate(tokens):
        if token in v4.TOLAK or token in v4._NEUTRAL_WORDS:
            continue
        negated = any(item in tokens[max(0, index - 3):index] for item in v4.NEGASI)
        if token in v4.POSITIF:
            score += -1 if negated else 1
        elif token in v4.NEGATIF:
            score += 1 if negated else -1
    if score > 0:
        return "positif"
    if score < 0:
        return "negatif"
    try:
        if float(rating) >= 5:
            return "positif"
        if float(rating) <= 1:
            return "negatif"
    except (TypeError, ValueError):
        pass
    return "netral"


def label_from_raw(
    raw_df: pd.DataFrame,
    progress: Callable | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Buat label v5 dari data mentah tanpa membaca label v4 atau baseline."""
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
        for sentence in v4._SENTENCE_SPLIT.split(review_text):
            sentence = sentence.strip(" ;:!?\"'“”‘’")
            if not sentence:
                continue
            for clause in v4._split_clauses(sentence):
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
        raise ValueError("Eksperimen v5 tidak menghasilkan label")
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
