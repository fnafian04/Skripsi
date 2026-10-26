"""Preprocessing dan pelabelan ulasan mentah Nafinder."""
from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd

from .text_processing import TextPreprocessor, potong_template, proses_ulasan

REQUIRED_RAW = {
    "placeId", "nama_kafe", "teks_ulasan", "rating_ulasan",
    "tanggal_ulasan", "kafe_id", "valid",
}

_PENANDA_TEMPLATE = re.compile(
    r"(?:Jenis pesanan|Jenis makanan|Jenis tempat|Tarif per orang|Waktu tunggu|"
    r"Makanan:\s*\d|Layanan:\s*\d|Suasana:\s*\d|Tingkat kebisingan|"
    r"Tempat parkir|Opsi tempat parkir|Jenis tempat duduk|Kesesuaian|Sumber daya|"
    r"Fasilitas untuk|Aksesibilitas|Layanan pelanggan|Pilihan makanan|Kategori)",
    re.I,
)


def _potong_template(teks: str) -> str:
    """Potong boilerplate Google Maps dari akhir teks ulasan."""
    teks = re.sub(r"\s+", " ", str(teks)).strip()
    match = _PENANDA_TEMPLATE.search(teks)
    return teks[:match.start()].strip() if match else teks


def preprocess_and_label(
    raw_df: pd.DataFrame,
    progress: Callable | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Ubah CSV mentah menjadi data berlabel aspek dan sentimen."""
    missing = REQUIRED_RAW - set(raw_df.columns)
    if missing:
        raise ValueError(f"Kolom CSV mentah tidak lengkap: {', '.join(sorted(missing))}")

    df = raw_df.copy().reset_index(drop=True)
    df["review_id"] = np.arange(len(df), dtype=np.int64)
    df["teks_ulasan"] = df["teks_ulasan"].fillna("").astype(str).map(_potong_template)
    df["valid"] = df["valid"].astype(str).str.lower().eq("true")

    n_raw = len(df)
    df = df[df["valid"]].copy().reset_index(drop=True)
    n_valid = len(df)

    preprocessor = TextPreprocessor()
    out = []
    for i, row in enumerate(df.itertuples(index=False), 1):
        labels = proses_ulasan(row.teks_ulasan, row.rating_ulasan, preprocessor)
        base = {
            "review_id": int(row.review_id),
            "placeId": row.placeId,
            "nama_kafe": row.nama_kafe,
            "kafe_id": row.kafe_id,
            "rating_ulasan": row.rating_ulasan,
            "asal_sub_kalimat": potong_template(row.teks_ulasan),
        }
        for item in labels:
            out.append({
                **base,
                "sub_kalimat": item["sub_kalimat"],
                "aspek": item["aspek"],
                "sentimen": item["sentimen"],
                "teks_preprocessed": item["teks_preprocessed"],
            })

        if progress and (i in (25, 100, 250, 500) or i % 500 == 0 or i == n_valid):
            progress(i, n_valid, len(out))

    labeled = pd.DataFrame(out)
    if labeled.empty:
        raise ValueError("Tidak ada label yang dihasilkan dari ulasan valid.")

    summary = {
        "ulasan_awal": n_raw,
        "ulasan_valid": n_valid,
        "label_setelah_preprocessing": len(labeled),
        "jumlah_review_berlabel": int(labeled.review_id.nunique()),
        "jumlah_kafe": int(labeled.placeId.nunique()),
        "distribusi_sentimen": labeled.sentimen.value_counts().to_dict(),
    }
    return labeled, summary
