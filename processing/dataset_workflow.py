"""Pengelola workflow multi-dataset Nafinder: simpan, aktifkan, preprocessing, klasifikasi, hapus.

Tanggung jawab modul ini:
- list_datasets     : mendata seluruh berkas dataset yang tersedia di storage.
- save_dataset      : menyimpan file CSV baru ke repositori multi-dataset (tanpa menghapus file lama).
- activate_dataset  : menentukan dataset aktif yang sedang digunakan dalam pipeline.
- run_preprocessing : memanggil pipeline.preprocess_and_label, menyimpan ulasan & label ke DB.
- run_classification: memanggil pipeline.fit_evaluate_manual, menyimpan metrik ke DB.
- delete_dataset    : menghapus berkas dataset tertentu atau membersihkan data pipeline.

Aturan penting:
- Berkas CSV mentah TIDAK BOLEH dihapus setelah preprocessing selesai agar pengguna
  dapat melakukan preprocessing ulang sewaktu-waktu.
- Penghapusan satu dataset tidak merusak dataset lainnya atau menimbulkan error di sistem.
"""
from __future__ import annotations
import datetime
import json
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import insert, text
from werkzeug.utils import secure_filename

from db import engine
from models.tables import EvaluasiModel, LabelSentimen, Ulasan
from processing.classification import fit_evaluate_manual
from processing.preprocessing import preprocess_and_label, REQUIRED_RAW
from processing.text_processing import (
    TextPreprocessor, deteksi_aspek, pecah_subkalimat, potong_template,
)
from processing.workflow_state import (
    progress_callback, read_state, reset_state, write_state
)

BASE = Path(__file__).resolve().parents[1]
IS_SERVERLESS = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))

if IS_SERVERLESS or not os.access(BASE, os.W_OK):
    import tempfile
    UPLOAD = Path(tempfile.gettempdir()) / 'nafinder_uploads'
else:
    UPLOAD = BASE / 'uploads_tmp'

try:
    UPLOAD.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

DATASETS_DIR = UPLOAD / 'datasets'
try:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

REGISTRY_FILE = UPLOAD / '.datasets_registry.json'
CACHE = UPLOAD / '.labels_cache.pkl'
SOURCE = UPLOAD / '.dataset_source.csv'

# Path model dan vectorizer pre-trained
ASSETS_DIR = Path(__file__).resolve().parent / 'assets'
SAVED_MODEL_FILE = ASSETS_DIR / 'model_mnb.pkl'
SAVED_VEC_FILE = ASSETS_DIR / 'vectorizer.pkl'

_RUNTIME_MODEL = None
_RUNTIME_VECTORIZER = None


def _migrate_datasets_once():
    """Pindahkan berkas CSV lama dari uploads_tmp langsung ke uploads_tmp/datasets/ jika belum ada."""
    try:
        for name in ('dataset_semprog.csv',):
            src = UPLOAD / name
            dst = DATASETS_DIR / name
            if src.is_file():
                if not dst.exists():
                    shutil.move(str(src), str(dst))
                else:
                    src.unlink(missing_ok=True)
    except OSError:
        pass


_migrate_datasets_once()


def _read_registry() -> dict[str, Any]:
    for p in (REGISTRY_FILE, BASE / 'uploads_tmp' / '.datasets_registry.json'):
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                pass
    return {}


def _write_registry(data: dict[str, Any]):
    try:
        REGISTRY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    except OSError:
        pass


def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    return f"{size_bytes / 1024:.1f} KB"


def get_dataset_path(filename: str | None = None) -> Path | None:
    """Cari path berkas dataset berdasarkan nama atau dataset aktif saat ini."""
    if not filename:
        state = read_state()
        filename = state.get('dataset_name')

    if not filename:
        return None

    fname = Path(filename).name
    candidates = [
        DATASETS_DIR / fname,
        UPLOAD / fname,
        BASE / 'uploads_tmp' / 'datasets' / fname,
        BASE / 'uploads_tmp' / fname,
        SOURCE if fname == '.dataset_source.csv' else None
    ]
    for c in candidates:
        if c and c.is_file():
            return c
    return None


def dataset_aktif_path() -> Path | None:
    """Kembalikan path dataset aktif saat ini, atau None jika belum ada."""
    return get_dataset_path()


def list_datasets() -> list[dict[str, Any]]:
    """Daftar seluruh file dataset CSV yang ada di folder repositori datasets."""
    reg = _read_registry()
    state = read_state()
    active_name = state.get('dataset_name')

    datasets = []
    reg_changed = False

    seen_names: set[str] = set()
    csv_files: list[Path] = []
    for folder in (DATASETS_DIR, BASE / 'uploads_tmp' / 'datasets'):
        if folder.is_dir():
            for p in sorted(folder.glob('*.csv'), key=lambda fp: fp.stat().st_mtime, reverse=True):
                if p.name not in seen_names:
                    seen_names.add(p.name)
                    csv_files.append(p)

    for p in csv_files:
        name = p.name
        stat = p.stat()
        meta = reg.get(name, {})

        # Hitung jumlah baris jika belum tercatat di registry
        if 'row_count' not in meta or meta.get('size_bytes') != stat.st_size:
            try:
                with p.open('r', encoding='utf-8', errors='ignore') as f:
                    meta['row_count'] = max(0, sum(1 for _ in f) - 1)
            except Exception:
                meta['row_count'] = 0
            meta['size_bytes'] = stat.st_size
            meta['uploaded_at'] = datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
            reg[name] = meta
            reg_changed = True

        cache_file = UPLOAD / f'.labels_{name}.pkl'
        bundled_cache = BASE / 'uploads_tmp' / f'.labels_{name}.pkl'
        has_labels = cache_file.exists() or bundled_cache.exists()
        is_active = (name == active_name)

        datasets.append({
            'name': name,
            'size_bytes': stat.st_size,
            'size_formatted': _format_size(stat.st_size),
            'uploaded_at': meta.get('uploaded_at', '-'),
            'row_count': meta.get('row_count', 0),
            'is_active': is_active,
            'has_labels': has_labels,
            'preprocessed': meta.get('preprocessed', False) or has_labels or (is_active and state.get('preprocessed', False)),
            'classified': meta.get('classified', False) or (is_active and state.get('classified', False)),
        })

    if reg_changed:
        _write_registry(reg)

    return datasets


def activate_dataset(filename: str) -> dict:
    """Aktifkan salah satu dataset dari repositori datasets."""
    p = get_dataset_path(filename)
    if not p or not p.is_file():
        raise FileNotFoundError(f"Dataset '{filename}' tidak ditemukan di repositori.")

    fname = p.name
    reg = _read_registry()
    meta = reg.get(fname, {})

    cache_file = UPLOAD / f'.labels_{fname}.pkl'
    bundled_cache = BASE / 'uploads_tmp' / f'.labels_{fname}.pkl'
    src_cache = cache_file if cache_file.exists() else (bundled_cache if bundled_cache.exists() else None)
    has_labels = src_cache is not None
    if src_cache and src_cache != CACHE:
        try:
            shutil.copy2(src_cache, CACHE)
        except OSError:
            pass

    # Sinkronisasi ke staging SOURCE untuk kompatibilitas script lama
    try:
        shutil.copy2(p, SOURCE)
    except Exception:
        pass

    is_prep = bool(has_labels or meta.get('preprocessed', False))
    is_cls = bool(meta.get('classified', False))
    reset_state(dataset_name=fname, dataset_size_bytes=p.stat().st_size)
    write_state(
        preprocessed=is_prep,
        classified=is_cls,
        percent=100 if is_cls else (60 if is_prep else 0),
        checkpoint=6 if is_prep else 0,
        checkpoint_title='Preprocessing selesai' if is_prep else 'Belum berjalan',
        checkpoint_action='Siap untuk klasifikasi atau preprocessing ulang' if is_prep else 'Klik Jalankan Preprocessing untuk memulai',
        class_checkpoint=5 if is_cls else 0,
        class_checkpoint_title='Klasifikasi selesai' if is_cls else 'Belum berjalan',
        class_checkpoint_action='Model selesai dievaluasi' if is_cls else ('Siap dijalankan' if is_prep else 'Menunggu preprocessing selesai'),
        message=f"Dataset '{fname}' diaktifkan — " + (
            "siap untuk klasifikasi atau preprocessing ulang"
            if is_prep else "jalankan tahap preprocessing terlebih dahulu"
        )
    )
    return read_state()


def save_dataset(source, original_name: str | None = None, set_active: bool = True) -> Path:
    """Simpan file CSV baru ke repositori uploads_tmp/datasets/ tanpa menimpa dataset lain.

    Args:
        source: Path atau file objek Flask (FileStorage).
        original_name: Nama asli file yang diunggah.
        set_active: Jika True, langsung jadikan dataset ini sebagai dataset aktif.
    """
    clean_name = secure_filename(Path(original_name or 'dataset.csv').name)
    if not clean_name or not clean_name.lower().endswith('.csv'):
        raise ValueError('Berkas yang diunggah harus berupa file CSV.')

    target = DATASETS_DIR / clean_name
    if target.exists():
        raise ValueError(f"Dataset '{clean_name}' sudah ada. Gunakan nama file lain.")

    try:
        if isinstance(source, (str, Path)):
            shutil.copy2(source, target)
        else:
            source.save(target)
    except OSError:
        target.unlink(missing_ok=True)
        raise

    try:
        sample = pd.read_csv(target, nrows=1, encoding='utf-8-sig')
        if sample.empty:
            raise ValueError('CSV harus memiliki setidaknya satu baris data.')
        missing = REQUIRED_RAW - set(sample.columns)
        if missing:
            raise ValueError(
                f"Kolom CSV tidak lengkap: {', '.join(sorted(missing))}"
            )
    except (pd.errors.EmptyDataError, pd.errors.ParserError,
            UnicodeDecodeError, OSError) as e:
        target.unlink(missing_ok=True)
        raise ValueError(f"Berkas bukan CSV yang valid atau rusak: {e}")
    except ValueError:
        target.unlink(missing_ok=True)
        raise

    # Catat ke registry
    reg = _read_registry()
    stat = target.stat()
    try:
        with target.open('r', encoding='utf-8', errors='ignore') as f:
            row_count = max(0, sum(1 for _ in f) - 1)
    except Exception:
        row_count = 0

    # Bersihkan cache label lama jika berkas ini diunggah ulang / ditimpa
    old_cache = UPLOAD / f'.labels_{clean_name}.pkl'
    if old_cache.exists():
        old_cache.unlink(missing_ok=True)

    reg[clean_name] = {
        'size_bytes': stat.st_size,
        'uploaded_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'row_count': row_count,
        'preprocessed': False,
        'classified': False,
    }
    _write_registry(reg)

    if set_active or not read_state().get('dataset_name'):
        activate_dataset(clean_name)

    return target


def _get_cafe_upsert_sql():
    """Query upsert kafe master kompatibel PostgreSQL dan MySQL."""
    if engine.dialect.name == 'postgresql':
        return text("""
            INSERT INTO kafe(place_id,nama,alamat,rating_gmaps,jumlah_ulasan,buka_24jam,wifi,stopkontak,meja_kerja)
            VALUES(:place,:name,:address,:rating,:count,:hours,:wifi,:outlet,:desk)
            ON CONFLICT (place_id) DO UPDATE SET
              nama=EXCLUDED.nama, alamat=EXCLUDED.alamat,
              rating_gmaps=EXCLUDED.rating_gmaps, jumlah_ulasan=EXCLUDED.jumlah_ulasan,
              buka_24jam=EXCLUDED.buka_24jam, wifi=EXCLUDED.wifi,
              stopkontak=EXCLUDED.stopkontak, meja_kerja=EXCLUDED.meja_kerja
        """)
    return text("""
            INSERT INTO kafe(place_id,nama,alamat,rating_gmaps,jumlah_ulasan,buka_24jam,wifi,stopkontak,meja_kerja)
            VALUES(:place,:name,:address,:rating,:count,:hours,:wifi,:outlet,:desk)
            ON DUPLICATE KEY UPDATE
              nama=VALUES(nama), alamat=VALUES(alamat),
              rating_gmaps=VALUES(rating_gmaps), jumlah_ulasan=VALUES(jumlah_ulasan),
              buka_24jam=VALUES(buka_24jam), wifi=VALUES(wifi),
              stopkontak=VALUES(stopkontak), meja_kerja=VALUES(meja_kerja)
        """)


def _get_skor_aspek_sql(include_netral: bool = True):
    """Query agregasi skor aspek per kafe kompatibel ANSI SQL (PostgreSQL & MySQL)."""
    netral_expr = "SUM(CASE WHEN sentimen='netral' THEN 1 ELSE 0 END)" if include_netral else "0"
    return text(f"""
            INSERT INTO skor_aspek(kafe_id,aspek,jumlah_positif,jumlah_netral,jumlah_negatif,
                                   persen_positif,persen_negatif,label_kekuatan,label_kelemahan)
            SELECT kafe_id, aspek,
              SUM(CASE WHEN sentimen='positif' THEN 1 ELSE 0 END),
              {netral_expr},
              SUM(CASE WHEN sentimen='negatif' THEN 1 ELSE 0 END),
              ROUND(100.0 * SUM(CASE WHEN sentimen='positif' THEN 1 ELSE 0 END) / COUNT(*), 2),
              ROUND(100.0 * SUM(CASE WHEN sentimen='negatif' THEN 1 ELSE 0 END) / COUNT(*), 2),
              CASE WHEN (100.0 * SUM(CASE WHEN sentimen='positif' THEN 1 ELSE 0 END) / COUNT(*)) >= 60 THEN TRUE ELSE FALSE END,
              CASE WHEN (100.0 * SUM(CASE WHEN sentimen='negatif' THEN 1 ELSE 0 END) / COUNT(*)) >= 30 THEN TRUE ELSE FALSE END
            FROM label_sentimen GROUP BY kafe_id, aspek
        """)


def _store_preprocessed_labels(labels: pd.DataFrame) -> tuple[int, int, int]:
    """Simpan CSV hasil preprocessing langsung ke database."""
    df = labels.copy()
    required = {'review_id', 'placeId', 'nama_kafe', 'sub_kalimat',
                'aspek', 'sentimen', 'teks_preprocessed'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Kolom CSV hasil preprocessing tidak lengkap: {', '.join(sorted(missing))}"
        )

    df['sentimen'] = df['sentimen'].astype(str).str.lower().str.strip()
    df = df[df['sentimen'].isin(['positif', 'negatif'])].copy()
    if df.empty:
        raise ValueError('Tidak ada label positif atau negatif pada CSV hasil preprocessing.')

    cafe_rows = []
    for place_id, group in df.groupby('placeId', sort=False):
        first = group.iloc[0]
        blob = ' '.join(
            group['asal_sub_kalimat'].fillna('').astype(str).tolist()
            if 'asal_sub_kalimat' in group.columns
            else group['sub_kalimat'].fillna('').astype(str).tolist()
        ).lower()
        address = str(first.get('kafe_id', '')).split('|', 1)[-1].strip()[:255]
        rates = pd.to_numeric(group['rating_ulasan'], errors='coerce').dropna() \
            if 'rating_ulasan' in group.columns else pd.Series(dtype='float64')
        cafe_rows.append({
            'place': str(place_id),
            'name': str(first.get('nama_kafe', 'Kafe'))[:160],
            'address': address,
            'rating': round(float(rates.mean()), 1) if len(rates) else None,
            'count': int(group['review_id'].nunique()),
            'hours': bool('24 jam' in blob or '24jam' in blob),
            'wifi': bool('wifi' in blob or 'wi-fi' in blob or 'internet' in blob),
            'outlet': bool(any(q in blob for q in ('colokan', 'stopkontak', 'charger'))),
            'desk': bool(any(q in blob for q in ('meja', 'kursi', 'nugas', 'laptop', 'wfc', 'kerja'))),
        })

    with engine.begin() as conn:
        conn.execute(_get_cafe_upsert_sql(), cafe_rows)

        ids = {p: int(i) for p, i in conn.execute(text('SELECT place_id, id FROM kafe')).fetchall()}
        conn.execute(text('DELETE FROM label_sentimen'))
        conn.execute(text('DELETE FROM skor_aspek'))
        conn.execute(text('DELETE FROM ulasan'))

        review_rows = []
        for _, r in df.drop_duplicates(['placeId', 'review_id']).iterrows():
            source_text = str(r.get('asal_sub_kalimat', r['sub_kalimat']))
            rating = pd.to_numeric(pd.Series([r.get('rating_ulasan')]), errors='coerce').iloc[0]
            rating_value = None
            if pd.notna(rating):
                rating_value = max(1, min(5, int(round(float(rating)))))
            review_rows.append({
                'kafe_id': ids[str(r.placeId)],
                'review_id': int(r.review_id),
                'teks': source_text[:60000],
                'rating': rating_value,
                'tanggal': None,
                'teks_preprocessed': str(r.teks_preprocessed)[:60000],
                'valid': True,
            })

        ins_review = insert(Ulasan)
        for i in range(0, len(review_rows), 1000):
            conn.execute(ins_review, review_rows[i:i + 1000])

        uid = {
            (int(k), int(r)): int(i)
            for i, k, r in conn.execute(text('SELECT id,kafe_id,review_id FROM ulasan')).fetchall()
        }

        label_rows = []
        for _, r in df.iterrows():
            key = (ids.get(str(r.placeId)), int(r.review_id))
            ulasan_id = uid.get(key)
            if ulasan_id:
                label_rows.append({
                    'ulasan_id': ulasan_id,
                    'kafe_id': key[0],
                    'sub_kalimat': str(r.sub_kalimat)[:512],
                    'aspek': r.aspek,
                    'sentimen': r.sentimen,
                })

        ins_label = insert(LabelSentimen)
        for i in range(0, len(label_rows), 1500):
            conn.execute(ins_label, label_rows[i:i + 1500])

        conn.execute(_get_skor_aspek_sql(include_netral=False))

    return len(review_rows), len(label_rows), len(cafe_rows)


def _store_labels(raw: pd.DataFrame, labels: pd.DataFrame, user_id=None) -> tuple[int, int, int]:
    """Simpan label preprocessing ke database: kafe, ulasan, label_sentimen, skor_aspek."""
    df_raw = raw.copy()
    df_raw['review_id'] = range(len(df_raw))
    df_raw['valid'] = df_raw.valid.astype(str).str.lower().eq('true')
    valid = df_raw[df_raw.valid].copy()

    # Kumpulkan data master kafe dari ulasan valid
    cafes = []
    for place, group in valid.groupby('placeId', sort=False):
        first = group.iloc[0]
        blob = ' '.join(group.teks_ulasan.fillna('').astype(str)).lower()
        rates = pd.to_numeric(group.rating_ulasan, errors='coerce').dropna()
        addr = str(first.get('kafe_id', '')).split('|', 1)[-1].strip()[:255] if pd.notna(first.get('kafe_id')) else ''
        cafes.append({
            'place': str(place),
            'name': str(first.get('nama_kafe', 'Kafe'))[:160],
            'address': addr,
            'rating': round(float(rates.mean()), 1) if len(rates) else None,
            'count': int(group.review_id.nunique()),
            'hours': bool('24 jam' in blob or '24jam' in blob),
            'wifi': bool('wifi' in blob or 'wi-fi' in blob or 'internet' in blob),
            'outlet': bool(any(q in blob for q in ('colokan', 'stopkontak', 'charger'))),
            'desk': bool(any(q in blob for q in ('meja', 'kursi', 'nugas', 'laptop', 'wfc', 'kerja'))),
        })

    with engine.begin() as conn:
        # Upsert kafe master
        conn.execute(_get_cafe_upsert_sql(), cafes)

        ids = {p: int(i) for p, i in conn.execute(text('SELECT place_id, id FROM kafe')).fetchall()}

        # Hapus ulasan dan label dataset lama di database sebelum mengisi dataset yang baru aktif
        conn.execute(text('DELETE FROM label_sentimen'))
        conn.execute(text('DELETE FROM skor_aspek'))
        conn.execute(text('DELETE FROM ulasan'))

        # Simpan ulasan mentah
        review = []
        for _, r in valid.drop_duplicates(['placeId', 'review_id']).iterrows():
            try:
                rating = max(1, min(5, int(round(float(r.rating_ulasan)))))
            except (ValueError, TypeError):
                rating = None
            review.append({
                'kafe_id': ids[str(r.placeId)],
                'review_id': int(r.review_id),
                'teks': str(r.teks_ulasan)[:60000],
                'rating': rating,
                'tanggal': None,
                'teks_preprocessed': None,
                'valid': True,
            })

        ins = insert(Ulasan)
        for i in range(0, len(review), 1000):
            conn.execute(ins, review[i:i + 1000])

        # Peta (kafe_id, review_id) -> ulasan.id
        uid = {
            (int(k), int(r)): int(i)
            for i, k, r in conn.execute(text('SELECT id,kafe_id,review_id FROM ulasan')).fetchall()
        }

        # Simpan label aspek-sentimen
        lr = []
        for _, r in labels.iterrows():
            key = (ids.get(str(r.placeId)), int(r.review_id))
            if key[0] is not None:
                ulid = uid.get(key)
                if ulid:
                    lr.append({
                        'ulasan_id': ulid,
                        'kafe_id': key[0],
                        'sub_kalimat': str(r.sub_kalimat)[:512],
                        'aspek': r.aspek,
                        'sentimen': r.sentimen,
                    })

        inslabel = insert(LabelSentimen)
        for i in range(0, len(lr), 1500):
            conn.execute(inslabel, lr[i:i + 1500])

        # Hitung agregasi skor_aspek per kafe
        conn.execute(_get_skor_aspek_sql(include_netral=True))

    return len(review), len(lr), len(cafes)


def run_preprocessing(filename: str | None = None, user_id=None) -> dict:
    """Jalankan preprocessing dan pelabelan, simpan hasilnya ke DB dan cache."""
    global _RUNTIME_MODEL, _RUNTIME_VECTORIZER
    _RUNTIME_MODEL = None
    _RUNTIME_VECTORIZER = None
    csv_path = get_dataset_path(filename)
    if not csv_path or not csv_path.is_file():
        raise FileNotFoundError("Berkas dataset tidak ditemukan. Pilih atau unggah dataset terlebih dahulu.")

    active_name = csv_path.name
    write_state(
        dataset_name=active_name,
        dataset_size_bytes=csv_path.stat().st_size,
        preprocessed=False,
        classified=False,
        percent=12,
        checkpoint=1,
        checkpoint_title='Persiapan data',
        checkpoint_action='Membuka dan membaca berkas CSV ulasan',
        checkpoint_result=f'Membaca berkas {active_name}',
        message=f'Tahap 1/6: Membaca berkas {active_name}...'
    )
    time.sleep(1.0)

    raw = pd.read_csv(csv_path, encoding='utf-8-sig')
    preprocessed_columns = {'review_id', 'placeId', 'nama_kafe', 'sub_kalimat',
                            'aspek', 'sentimen', 'teks_preprocessed'}
    if preprocessed_columns.issubset(set(raw.columns)):
        labels = raw.copy()
        labels['sentimen'] = labels['sentimen'].astype(str).str.lower().str.strip()
        labels = labels[labels['sentimen'].isin(['positif', 'negatif'])].copy()
        if labels.empty:
            raise ValueError('Tidak ada label positif atau negatif pada CSV hasil preprocessing.')

        write_state(
            percent=20,
            checkpoint=1,
            checkpoint_title='Persiapan data',
            checkpoint_action='Memvalidasi struktur kolom dan memuat baris data',
            checkpoint_result=f'{len(raw):,} baris data berhasil dimuat dari {active_name}',
            message=f'Tahap 1/6: Validasi struktur data ({len(raw):,} baris)',
        )
        time.sleep(1.0)

        write_state(
            percent=35,
            checkpoint=2,
            checkpoint_title='Potong template',
            checkpoint_action='Membersihkan teks boilerplate terjemahan Google Maps',
            checkpoint_result='Teks ulasan bersih dari template bawaan Google Maps',
            message='Tahap 2/6: Membersihkan boilerplate template ulasan...',
        )
        time.sleep(1.0)

        write_state(
            percent=52,
            checkpoint=3,
            checkpoint_title='Pecah dan beri label',
            checkpoint_action='Memecah sub-kalimat serta memetakan 6 aspek WFC dan polaritas sentimen',
            checkpoint_result=f'{len(labels):,} sub-kalimat berlabel positif/negatif tervalidasi',
            message=f'Tahap 3/6: Pemecahan sub-kalimat & pemetaan {len(labels):,} label...',
        )
        time.sleep(1.0)

        write_state(
            percent=70,
            checkpoint=4,
            checkpoint_title='Enam tahap preprocessing',
            checkpoint_action='Menjalankan Cleaning, Case Folding, Tokenizing, Normalisasi, Stopword Removal, dan Stemming Sastrawi',
            checkpoint_result=f'Transformasi 6 tahap teks pada {len(labels):,} data terverifikasi',
            message='Tahap 4/6: Cleaning, Case Folding, Tokenizing, Normalisasi, Stopword & Stemming...',
        )
        time.sleep(1.0)

        write_state(
            percent=86,
            checkpoint=5,
            checkpoint_title='Simpan hasil',
            checkpoint_action='Menyimpan kafe, ulasan, label sentimen, dan agregasi skor aspek ke database',
            checkpoint_result='Menyinkronkan dataset ke database aktif...',
            message='Tahap 5/6: Menyimpan hasil preprocessing ke database...',
        )
        dataset_cache = UPLOAD / f'.labels_{active_name}.pkl'
        labels.to_pickle(dataset_cache)
        labels.to_pickle(CACHE)

        counts = _store_preprocessed_labels(labels)
        time.sleep(1.0)

        reg = _read_registry()
        meta = reg.get(active_name, {})
        meta['size_bytes'] = csv_path.stat().st_size
        meta['row_count'] = len(raw)
        meta['preprocessed'] = True
        meta['classified'] = False
        meta['labels_count'] = len(labels)
        reg[active_name] = meta
        _write_registry(reg)

        write_state(
            preprocessed=True,
            classified=False,
            percent=100,
            checkpoint=6,
            checkpoint_title='Preprocessing selesai',
            checkpoint_action='Semua tahap preprocessing selesai — data siap diklasifikasikan',
            checkpoint_result=f'{counts[2]:,} kafe, {counts[0]:,} ulasan, dan {counts[1]:,} label tersimpan',
            message=f'Preprocessing selesai: {len(labels):,} label siap diklasifikasikan',
        )
        return {
            'labels': len(labels),
            'reviews': int(labels.review_id.nunique()),
            'cafes': counts[2],
            'replayed_preprocessing': True,
        }

    write_state(
        percent=18,
        checkpoint=1,
        checkpoint_title='Persiapan data',
        checkpoint_action='Membaca CSV dan memeriksa kolom wajib',
        checkpoint_result=f'{len(raw):,} baris mentah ditemukan pada {active_name}',
        message=f'Tahap 1/6 — {len(raw):,} ulasan siap diproses',
    )
    time.sleep(1.0)

    def _progress(done: int, total: int, n_labels: int):
        pct = min(84, 45 + int(39 * done / total)) if total else 60
        write_state(
            percent=pct,
            checkpoint=4,
            checkpoint_title='Enam tahap preprocessing',
            checkpoint_action='Menormalisasi, tokenisasi, menghapus stopword, dan stemming',
            checkpoint_result=f'{n_labels:,} label sementara dari {done:,} ulasan selesai',
            message=f'Tahap 4/6: Preprocessing & pelabelan ({done:,}/{total:,} ulasan)',
        )

    write_state(
        percent=32,
        checkpoint=2,
        checkpoint_title='Potong template',
        checkpoint_action='Menghapus bagian boilerplate Google Maps dari teks ulasan',
        checkpoint_result='Teks ulasan siap dibaca sebagai isi review',
        message='Tahap 2/6: Memotong template Google Maps...',
    )
    time.sleep(1.0)

    write_state(
        percent=45,
        checkpoint=3,
        checkpoint_title='Pecah dan beri label',
        checkpoint_action='Memecah sub-kalimat lalu mendeteksi aspek dan sentimen lexicon',
        checkpoint_result='Label aspek-sentimen dikumpulkan per ulasan',
        message='Tahap 3/6: Memecah sub-kalimat dan pelabelan aspek...',
    )
    time.sleep(1.0)

    labels, summary = preprocess_and_label(raw, progress=_progress)

    write_state(
        percent=88,
        checkpoint=5,
        checkpoint_title='Simpan hasil',
        checkpoint_action='Menyimpan ulasan, label, dan agregasi skor aspek ke database',
        checkpoint_result=f'{len(labels):,} label dari {summary["jumlah_review_berlabel"]:,} ulasan siap disimpan',
        message='Tahap 5/6 — menyimpan hasil preprocessing ke database',
    )
    time.sleep(1.0)

    # Simpan ke cache khusus dataset dan cache umum
    dataset_cache = UPLOAD / f'.labels_{active_name}.pkl'
    labels.to_pickle(dataset_cache)
    labels.to_pickle(CACHE)

    counts = _store_labels(raw, labels, user_id)

    # Update metadata di registry
    reg = _read_registry()
    meta = reg.get(active_name, {})
    meta['size_bytes'] = csv_path.stat().st_size
    meta['row_count'] = len(raw)
    meta['preprocessed'] = True
    meta['classified'] = False
    meta['labels_count'] = len(labels)
    reg[active_name] = meta
    _write_registry(reg)

    write_state(
        preprocessed=True,
        classified=False,
        message=(
            f"Preprocessing selesai: {len(labels):,} label "
            f"dari {summary['jumlah_review_berlabel']:,} ulasan"
        ),
        percent=100,
        checkpoint=6,
        checkpoint_title='Preprocessing selesai',
        checkpoint_action='Semua tahap preprocessing dan pelabelan telah selesai',
        checkpoint_result=f'{counts[2]:,} kafe, {counts[0]:,} ulasan, dan {counts[1]:,} label tersimpan',
    )
    return {
        'labels': len(labels),
        'reviews': summary['jumlah_review_berlabel'],
        'cafes': counts[2],
    }


def run_classification(user_id=None) -> dict:
    """Jalankan TF-IDF, oversampling, MNB, dan evaluasi; simpan metrik ke DB."""
    global _RUNTIME_MODEL, _RUNTIME_VECTORIZER
    state = read_state()
    active_name = state.get('dataset_name')

    labels = None
    candidates = [
        UPLOAD / f'.labels_{active_name}.pkl' if active_name else None,
        CACHE,
        BASE / 'uploads_tmp' / f'.labels_{active_name}.pkl' if active_name else None,
        BASE / 'uploads_tmp' / '.labels_cache.pkl',
    ]
    for c in candidates:
        if c and c.is_file():
            labels = pd.read_pickle(c)
            break

    if labels is None:
        raise ValueError('Hasil preprocessing belum ditemukan. Jalankan preprocessing sebelum klasifikasi.')

    write_state(
        classified=False,
        percent=10,
        class_checkpoint=0,
        class_checkpoint_title='Menyiapkan klasifikasi',
        class_checkpoint_action='Membaca data label hasil preprocessing',
        class_checkpoint_result=f'{len(labels):,} sampel siap dilatih dan diuji',
        message='Menyiapkan alur klasifikasi MNB...'
    )
    time.sleep(0.8)

    def _progress(stage: str, done: int, total: int, detail: str):
        actions = {
            1: "Membagi ulasan menjadi 80% latih dan 20% uji berbasis review_id (Group Shuffle Split anti-bocor)",
            2: "Mengekstraksi fitur teks menggunakan Sublinear TF, Smooth IDF, dan Normalisasi L2",
            3: "Menggunakan data latih asli tanpa Random Oversampling",
            4: "Menghitung log prior dan log likelihood kata per kelas dengan Laplace smoothing (alpha=0,1)",
            5: "Menghitung Confusion Matrix, Akurasi, Precision, Recall, Macro F1, dan Weighted F1 pada data uji"
        }
        pct = min(92, int((done / total) * 85) + 10) if total else 50
        write_state(
            percent=pct,
            class_checkpoint=done,
            class_checkpoint_title=stage,
            class_checkpoint_action=actions.get(done, stage),
            class_checkpoint_result=detail,
            message=f"Tahap {done}/{total}: {stage} — {detail}"
        )
        time.sleep(1.0)

    metrics, model, vectorizer, _ = fit_evaluate_manual(
        labels,
        test_size=0.2,
        random_state=42,
        alpha=0.1,
        min_df=2,
        ros_target_ratio=0.0,
        progress=_progress,
    )
    _RUNTIME_MODEL = model
    _RUNTIME_VECTORIZER = vectorizer

    try:
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        with open(SAVED_MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
        with open(SAVED_VEC_FILE, 'wb') as f:
            pickle.dump(vectorizer, f)
    except OSError:
        pass

    cm = {
        'classes': metrics['classes'],
        'matrix': metrics['matrix'],
        'per_class': metrics['per_class'],
    }

    with engine.begin() as conn:
        conn.execute(text('DELETE FROM evaluasi_model'))
        conn.execute(insert(EvaluasiModel), {
            'nama_model': 'Multinomial Naive Bayes',
            'accuracy': metrics['accuracy'],
            'precision_macro': metrics['macro']['precision'],
            'recall_macro': metrics['macro']['recall'],
            'f1_macro': metrics['macro']['f1'],
            'f1_weighted': metrics['weighted']['f1'],
            'jumlah_testing': metrics['jumlah_testing'],
            'confusion_matrix': cm,
        })
        conn.execute(text("""
            INSERT INTO riwayat_upload(nama_file,jumlah_baris,jumlah_kafe,status,pesan,user_id)
            VALUES(:file,:rows,:cafes,'berhasil',:msg,:uid)
        """), {
            'file': active_name or 'dataset_tidak_diketahui.csv',
            'rows': metrics['jumlah_label_training'] + metrics['jumlah_label_testing'],
            'cafes': 0,
            'msg': (
                f"Klasifikasi manual; "
                f"accuracy={metrics['accuracy']:.6f}; "
                f"macro-F1={metrics['macro']['f1']:.6f}"
            ),
            'uid': user_id,
        })

    # Update metadata di registry
    if active_name:
        reg = _read_registry()
        meta = reg.get(active_name, {})
        meta['preprocessed'] = True
        meta['classified'] = True
        meta['accuracy'] = round(float(metrics['accuracy']), 4)
        meta['f1_macro'] = round(float(metrics['macro']['f1']), 4)
        reg[active_name] = meta
        _write_registry(reg)

    write_state(
        classified=True,
        percent=100,
        class_checkpoint=5,
        class_checkpoint_title='Klasifikasi selesai',
        class_checkpoint_action='Semua 5 tahap klasifikasi MNB dan evaluasi metrik telah tuntas',
        class_checkpoint_result=f"Akurasi: {metrics['accuracy']:.2%}, Macro F1: {metrics['macro']['f1']:.4f}",
        message=(
            f"Klasifikasi selesai: accuracy {metrics['accuracy']:.2%}; "
            f"macro F1 {metrics['macro']['f1']:.4f}"
        ),
    )
    return metrics


def predict_review(text: str) -> list[dict[str, str]]:
    """Prediksi aspek dan sentimen untuk ulasan baru memakai model aktif."""
    global _RUNTIME_MODEL, _RUNTIME_VECTORIZER
    if _RUNTIME_MODEL is None or _RUNTIME_VECTORIZER is None:
        # 1. Coba muat model & vectorizer yang telah di-bundle untuk produksi
        if SAVED_MODEL_FILE.is_file() and SAVED_VEC_FILE.is_file():
            try:
                import pickle
                with open(SAVED_MODEL_FILE, 'rb') as f:
                    _RUNTIME_MODEL = pickle.load(f)
                with open(SAVED_VEC_FILE, 'rb') as f:
                    _RUNTIME_VECTORIZER = pickle.load(f)
            except Exception:
                _RUNTIME_MODEL = None
                _RUNTIME_VECTORIZER = None

        # 2. Fallback: latih dinamis dari dataset cache jika model bundle tidak tersedia
        if _RUNTIME_MODEL is None or _RUNTIME_VECTORIZER is None:
            active_name = read_state().get('dataset_name')
            labels = None
            candidates = [
                UPLOAD / f'.labels_{active_name}.pkl' if active_name else None,
                CACHE,
                BASE / 'uploads_tmp' / f'.labels_{active_name}.pkl' if active_name else None,
                BASE / 'uploads_tmp' / '.labels_cache.pkl',
                BASE / 'uploads_tmp' / '.labels_dataset_Nov_preprocessed.csv.pkl'
            ]
            for c in candidates:
                if c and c.is_file():
                    try:
                        labels = pd.read_pickle(c)
                        break
                    except Exception:
                        pass

            if labels is None:
                raise RuntimeError("Data model belum tersedia. Hubungi pengelola aplikasi.")
            _, _RUNTIME_MODEL, _RUNTIME_VECTORIZER, _ = fit_evaluate_manual(
                labels,
                test_size=0.2,
                random_state=42,
                alpha=0.1,
                min_df=2,
                ros_target_ratio=0.0,
            )

    cleaned = potong_template(text)
    preprocessor = TextPreprocessor()
    predictions: list[dict[str, str]] = []
    for sub in pecah_subkalimat(cleaned):
        aspects = deteksi_aspek(sub)
        processed = preprocessor.preprocess(sub)
        if not aspects or not processed:
            continue
        sentiment = str(_RUNTIME_MODEL.predict(
            _RUNTIME_VECTORIZER.transform([processed])
        )[0])
        predictions.extend(
            {"sub_kalimat": sub, "aspek": aspect, "sentimen": sentiment}
            for aspect in aspects
        )
    return predictions


def delete_dataset(filename: str | None = None) -> dict:
    """Hapus satu dataset spesifik dari repositori.

    Jika dataset yang dihapus adalah dataset yang sedang aktif:
    akan beralih ke dataset lain yang tersedia secara otomatis,
    atau mereset status jika tidak ada dataset lain.
    Dataset lain dan data master tidak akan rusak.
    """
    state = read_state()
    active_name = state.get('dataset_name')

    target_name = filename or active_name
    if not target_name:
        return reset_state()

    # 1. Hapus SEMUA kemungkinan berkas fisik CSV dan cache labelnya
    candidates_to_delete = [
        DATASETS_DIR / target_name,
        UPLOAD / target_name,
        UPLOAD / f'.labels_{target_name}.pkl',
    ]
    for f in candidates_to_delete:
        if f.exists():
            f.unlink(missing_ok=True)

    # 2. Hapus dari registry
    reg = _read_registry()
    if target_name in reg:
        del reg[target_name]
        _write_registry(reg)

    # 3. Jika dataset yang dihapus adalah yang sedang aktif
    if target_name == active_name:
        remaining = [f.name for f in DATASETS_DIR.glob('*.csv')]
        if remaining:
            # Otomatis alihkan ke dataset pertama yang tersisa
            activate_dataset(remaining[0])
            write_state(message=f"Dataset '{target_name}' berhasil dihapus, dataset aktif dialihkan ke '{remaining[0]}'")
        else:
            # Tidak ada dataset tersisa
            if CACHE.exists():
                CACHE.unlink(missing_ok=True)
            if SOURCE.exists():
                SOURCE.unlink(missing_ok=True)
            reset_state()
            write_state(message="Semua dataset telah dihapus, unggah dataset baru untuk memulai")

    return read_state()



def clear_pipeline_database() -> None:
    """Hapus hasil pemrosesan di database tanpa menghapus file CSV mentah di storage."""
    global _RUNTIME_MODEL, _RUNTIME_VECTORIZER
    _RUNTIME_MODEL = None
    _RUNTIME_VECTORIZER = None
    with engine.begin() as conn:
        conn.execute(text('DELETE FROM label_sentimen'))
        conn.execute(text('DELETE FROM skor_aspek'))
        conn.execute(text('DELETE FROM ulasan'))
        conn.execute(text('DELETE FROM evaluasi_model'))

    if CACHE.exists():
        CACHE.unlink(missing_ok=True)

    write_state(
        preprocessed=False,
        classified=False,
        status='siap',
        checkpoint=0,
        checkpoint_title='',
        checkpoint_action='',
        checkpoint_result='',
        percent=0,
        message='Data pipeline di database telah dibersihkan, dataset siap diproses ulang'
    )
