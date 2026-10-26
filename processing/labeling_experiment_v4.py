"""Eksperimen labeling v4 berdasarkan hasil validasi manual 100 sampel."""
from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd

from .labeling_experiment_v3 import _ASPECT_WORDS, _SENTENCE_SPLIT, _contains
from .text_processing import (
    NEGASI,
    NEGATIF,
    NEGATIF_FRASE,
    POSITIF,
    POSITIF_FRASE,
    TOLAK,
    TextPreprocessor,
    potong_template,
)

_WORD = re.compile(r"[a-zA-Z']+")
_COORDINATING = re.compile(r"\s+(?:dan|serta|&)\s+", re.I)
_SERVICE = (
    "diantar", "dilayani", "menunggu", "nunggu", "penyajian",
    "barista", "kasir", "pegawai", "karyawan", "pelayan", "service",
)
_WFC = (
    "nugas", "wfc", "wifi", "wi-fi", "colokan", "stopkontak", "laptop",
    "meet up", "meeting", "kerja", "work", "charger", "bawa laptop",
)
_PRICE = (
    "harga", "ramah kantong", "value for money", "buy1get1", "buy 1 get 1",
    "rupiah", "rp ", "idr", "40 ribu", "murah", "mahal", "terjangkau",
    "affordable", "worth it",
)
_FOOD = (
    "makanan", "minuman", "menu", "kopi", "rasa", "latte", "teh", "roti",
    "kue", "dessert", "nasi", "mie", "pastry", "cappuccino", "espresso",
)
_PLACE = (
    "tempat", "suasana", "ambience", "vibe", "lighting", "view", "nyaman",
    "adem", "ramai", "sepi", "luas", "sempit", "nongkrong",
)
_NEUTRAL_WORDS = {"lumayan", "standar", "biasa", "cukup"}


def _has_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(_contains(text, phrase) for phrase in phrases)


def _aspects(text: str) -> list[str]:
    lowered = text.lower()
    found = [
        aspect for aspect, words in _ASPECT_WORDS.items()
        if any(_contains(lowered, word) for word in words)
    ]
    service = _has_any(lowered, _SERVICE)
    wfc = _has_any(lowered, _WFC)
    price = _has_any(lowered, _PRICE)
    food = _has_any(lowered, _FOOD)
    place = _has_any(lowered, _PLACE) or "tempat" in lowered
    if "kenyamanan" in lowered:
        place = True

    if service and "pelayanan" not in found:
        found.append("pelayanan")
    if wfc and "fasilitas_wfc" not in found:
        found.append("fasilitas_wfc")
    if price and "harga" not in found:
        found.append("harga")
    if place and "suasana" not in found:
        found.append("suasana")

    # Kata evaluasi untuk tempat tidak otomatis berarti makanan
    explicit_food = _has_any(lowered, _FOOD)
    if place and not explicit_food and "makanan_minuman" in found:
        found.remove("makanan_minuman")
    # Keluhan pengantaran/penyajian adalah pelayanan, bukan aspek makanan
    if service and not _has_any(lowered, _FOOD):
        found = [item for item in found if item != "makanan_minuman"]
    # Crew/staff bukan suasana
    if _has_any(lowered, ("crew", "staff", "barista", "pegawai", "karyawan")):
        found = [item for item in found if item != "suasana"]
    if _has_any(lowered, ("dirty latte", "dirty coffee")):
        found = [item for item in found if item != "kebersihan"]
    # Jika hanya menyebut nama menu tanpa evaluasi, tetap pertahankan makanan
    if _has_any(lowered, ("makanan", "minuman", "menu", "kopi", "mie", "teh", "latte")) and (
        "makanan_minuman" not in found
    ):
        found.append("makanan_minuman")
    return list(dict.fromkeys(found))


def _split_clauses(text: str) -> list[str]:
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


def _sentiment(text: str, rating: object) -> str:
    lowered = text.lower()
    if re.search(r"\b(?:kenyamanan|rating|nilai)\s*[123]\s*(?:/|dari)\s*5", lowered):
        return "negatif"
    score = sum(1 for phrase in POSITIF_FRASE if _contains(lowered, phrase))
    score -= sum(1 for phrase in NEGATIF_FRASE if _contains(lowered, phrase))
    if _has_any(lowered, ("tidak tersedia", "tidak ada", "minim", "cukup lama", "terlalu", "agak aneh")):
        score -= 1
    informational = _has_any(lowered, ("lumayan", "standar", "biasa", "pesan mie"))
    if informational and not _has_any(
        lowered, ("sangat", "enak", "lezat", "buruk", "kecewa")
    ):
        return "netral"
    if informational and not (
        _has_any(lowered, ("sangat", "enak", "lezat", "buruk", "kecewa"))
    ):
        score = 0
    for index, token in enumerate(_WORD.findall(lowered)):
        if token in TOLAK or token in _NEUTRAL_WORDS:
            continue
        negated = any(item in _WORD.findall(lowered)[max(0, index - 3):index] for item in NEGASI)
        if token in POSITIF:
            score += -1 if negated else 1
        elif token in NEGATIF:
            score += 1 if negated else -1
    if _has_any(lowered, ("minim", "tidak tersedia", "cukup lama", "agak aneh")):
        return "negatif"
    if score > 0:
        return "positif"
    if score < 0:
        return "negatif"
    if informational:
        return "netral"
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
    """Buat label v4 hanya dari data mentah, tanpa membaca label sebelumnya."""
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
        for sentence in _SENTENCE_SPLIT.split(review_text):
            sentence = sentence.strip(" ;:!?\"'“”‘’")
            if not sentence:
                continue
            for clause in _split_clauses(sentence):
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
        raise ValueError("Eksperimen v4 tidak menghasilkan label")
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
