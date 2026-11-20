"""Preprocessing dan pelabelan ulasan mentah Nafinder.

Modul ini menerima CSV mentah, memeriksa struktur kolomnya, membersihkan
teks tambahan dari Google Maps, lalu mengambil ulasan yang valid. Setiap
ulasan diproses menjadi sub-kalimat, dikenali aspek yang dibahas, diberi
label sentimen berbasis lexicon, dan disimpan sebagai tabel siap klasifikasi.
Satu ulasan dapat menghasilkan beberapa baris karena dapat membahas beberapa
kalimat atau aspek.

Alur utama modul ini adalah:
1. memastikan CSV memiliki kolom yang dibutuhkan;
2. membersihkan boilerplate dari teks ulasan;
3. menyaring ulasan yang ditandai valid;
4. memecah ulasan menjadi sub-kalimat dan memberi label aspek-sentimen;
5. mengembalikan tabel label beserta ringkasan hasil preprocessing.

Modul ini tidak melatih model klasifikasi. Label aspek dan sentimen dibuat
oleh aturan lexicon di ``text_processing.py`` dan menjadi input untuk tahap
klasifikasi berikutnya.
"""
from __future__ import annotations

import re  # Regular expression untuk merapikan dan memotong teks.
from typing import Callable  # Tipe callback untuk laporan progress.

import numpy as np  # Membuat ID review numerik.
import pandas as pd  # Mengolah data CSV berbentuk tabel.

from .text_processing import TextPreprocessor, potong_template, proses_ulasan  # Mesin preprocessing dan pelabelan.

REQUIRED_RAW = {  # Kolom minimum yang wajib tersedia pada CSV mentah.
    "placeId", "nama_kafe", "teks_ulasan", "rating_ulasan",
    "tanggal_ulasan", "kafe_id", "valid",
}

_PENANDA_TEMPLATE = re.compile(  # Penanda awal boilerplate Google Maps.
    r"(?:Jenis pesanan|Jenis makanan|Jenis tempat|Tarif per orang|Waktu tunggu|"
    r"Makanan:\s*\d|Layanan:\s*\d|Suasana:\s*\d|Tingkat kebisingan|"
    r"Tempat parkir|Opsi tempat parkir|Jenis tempat duduk|Kesesuaian|Sumber daya|"
    r"Fasilitas untuk|Aksesibilitas|Layanan pelanggan|Pilihan makanan|Kategori)",
    re.I,
)


def _potong_template(teks: str) -> str:
    """Potong boilerplate Google Maps dari akhir teks ulasan.

    Google Maps dapat menambahkan informasi terstruktur seperti jenis pesanan
    atau fasilitas di bagian akhir review. Informasi tersebut bukan isi opini,
    sehingga dipisahkan sebelum teks diproses menjadi label.
    """
    teks = re.sub(r"\s+", " ", str(teks)).strip()  # Rapikan spasi dan ubah input menjadi teks.
    match = _PENANDA_TEMPLATE.search(teks)  # Cari bagian template yang bukan opini.
    return teks[:match.start()].strip() if match else teks  # Ambil opini sebelum template.


def preprocess_and_label(
    raw_df: pd.DataFrame,
    progress: Callable | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Ubah CSV mentah menjadi data berlabel aspek dan sentimen.

    Satu review dapat menghasilkan beberapa baris label karena sebuah review
    bisa memiliki beberapa sub-kalimat atau membahas beberapa aspek.
    """
    missing = REQUIRED_RAW - set(raw_df.columns)  # Temukan kolom wajib yang belum ada.
    if missing:
        raise ValueError(f"Kolom CSV mentah tidak lengkap: {', '.join(sorted(missing))}")

    # ID dibuat sebelum filtering agar setiap sub-kalimat tetap terhubung
    # dengan review asalnya saat pembagian data anti-kebocoran.
    df = raw_df.copy().reset_index(drop=True)  # Salin data agar input asli tidak berubah.
    df["review_id"] = np.arange(len(df), dtype=np.int64)  # ID menjaga hubungan sub-kalimat dengan review.
    df["teks_ulasan"] = df["teks_ulasan"].fillna("").astype(str).map(_potong_template)  # Bersihkan teks.
    df["valid"] = df["valid"].astype(str).str.lower().eq("true")  # Ubah penanda valid menjadi boolean.

    n_raw = len(df)  # Simpan jumlah ulasan sebelum penyaringan.
    # Hanya review valid yang boleh menghasilkan label untuk model.
    df = df[df["valid"]].copy().reset_index(drop=True)  # Hanya ulasan valid masuk pipeline.
    n_valid = len(df)  # Simpan jumlah ulasan yang lolos penyaringan.

    preprocessor = TextPreprocessor()  # Siapkan aturan cleaning, normalisasi, dan stemming.
    out = []  # Menampung hasil label setiap sub-kalimat/aspek.
    for i, row in enumerate(df.itertuples(index=False), 1):
        # proses_ulasan melakukan pemotongan sub-kalimat, deteksi aspek,
        # penentuan sentimen, dan preprocessing teks untuk setiap hasil label.
        labels = proses_ulasan(row.teks_ulasan, row.rating_ulasan, preprocessor)  # Hasilkan label.
        base = {  # Kolom identitas yang diwariskan ke setiap label review.
            "review_id": int(row.review_id),
            "placeId": row.placeId,
            "nama_kafe": row.nama_kafe,
            "kafe_id": row.kafe_id,
            "rating_ulasan": row.rating_ulasan,
            "asal_sub_kalimat": potong_template(row.teks_ulasan),
        }
        for item in labels:  # Satu item mewakili satu sub-kalimat/aspek.
            if item["sentimen"] not in {"positif", "negatif"}:
                continue
            out.append({  # Gabungkan identitas review dengan hasil proses.
                **base,
                "sub_kalimat": item["sub_kalimat"],
                "aspek": item["aspek"],
                "sentimen": item["sentimen"],
                "teks_preprocessed": item["teks_preprocessed"],
            })

        if progress and (i in (25, 100, 250, 500) or i % 500 == 0 or i == n_valid):  # Update UI berkala.
            progress(i, n_valid, len(out))

    labeled = pd.DataFrame(out)  # Ubah hasil list menjadi tabel terstruktur.
    if labeled.empty:
        raise ValueError("Tidak ada label yang dihasilkan dari ulasan valid.")

    summary = {  # Ringkasan untuk dashboard dan laporan pipeline.
        "ulasan_awal": n_raw,
        "ulasan_valid": n_valid,
        "label_setelah_preprocessing": len(labeled),
        "jumlah_review_berlabel": int(labeled.review_id.nunique()),
        "jumlah_kafe": int(labeled.placeId.nunique()),
        "distribusi_sentimen": labeled.sentimen.value_counts().to_dict(),
        "distribusi_aspek": labeled.aspek.value_counts().to_dict(),
    }
    return labeled, summary  # Kembalikan data siap klasifikasi dan ringkasannya.
