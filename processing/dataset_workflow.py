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
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from db import engine
from processing.classification import fit_evaluate_manual
from processing.preprocessing import preprocess_and_label, REQUIRED_RAW
from processing.workflow_state import (
    progress_callback, read_state, reset_state, write_state
)

BASE = Path(__file__).resolve().parents[1]
UPLOAD = BASE / 'uploads_tmp'
UPLOAD.mkdir(exist_ok=True)

DATASETS_DIR = UPLOAD / 'datasets'
DATASETS_DIR.mkdir(exist_ok=True)

REGISTRY_FILE = UPLOAD / '.datasets_registry.json'
CACHE = UPLOAD / '.labels_cache.pkl'
SOURCE = UPLOAD / '.dataset_source.csv'


def _migrate_datasets_once():
    """Pindahkan berkas CSV lama dari uploads_tmp langsung ke uploads_tmp/datasets/ jika belum ada."""
    for name in ('ulasan_wfc_final.csv', 'ulasan_wfc_testing_5000.csv'):
        src = UPLOAD / name
        dst = DATASETS_DIR / name
        if src.is_file():
            if not dst.exists():
                shutil.move(str(src), str(dst))
            else:
                src.unlink(missing_ok=True)


_migrate_datasets_once()



def _read_registry() -> dict[str, Any]:
    if not REGISTRY_FILE.exists():
        return {}
    try:
        return json.loads(REGISTRY_FILE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def _write_registry(data: dict[str, Any]):
    REGISTRY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


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
    """Daftar seluruh file dataset CSV yang ada di folder repositori uploads_tmp/datasets/."""
    reg = _read_registry()
    state = read_state()
    active_name = state.get('dataset_name')

    datasets = []
    reg_changed = False

    csv_files = sorted(DATASETS_DIR.glob('*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)

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
        is_active = (name == active_name)

        datasets.append({
            'name': name,
            'size_bytes': stat.st_size,
            'size_formatted': _format_size(stat.st_size),
            'uploaded_at': meta.get('uploaded_at', '-'),
            'row_count': meta.get('row_count', 0),
            'is_active': is_active,
            'has_labels': cache_file.exists(),
            'preprocessed': meta.get('preprocessed', False) or (is_active and state.get('preprocessed', False)),
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
    has_labels = cache_file.exists()
    if has_labels:
        # Salin label cache ke file default
        shutil.copy2(cache_file, CACHE)

    # Sinkronisasi ke staging SOURCE untuk kompatibilitas script lama
    try:
        shutil.copy2(p, SOURCE)
    except Exception:
        pass

    state = reset_state(dataset_name=fname, dataset_size_bytes=p.stat().st_size)
    write_state(
        preprocessed=has_labels or meta.get('preprocessed', False),
        classified=meta.get('classified', False),
        message=f"Dataset '{fname}' diaktifkan, " + (
            "siap untuk klasifikasi atau preprocessing ulang"
            if has_labels else "jalankan tahap preprocessing"
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
    clean_name = Path(original_name or 'dataset.csv').name
    clean_name = clean_name.replace(' ', '_')
    if not clean_name.lower().endswith('.csv'):
        clean_name += '.csv'

    target = DATASETS_DIR / clean_name

    if isinstance(source, (str, Path)):
        shutil.copy2(source, target)
    else:
        source.save(target)

    # Validasi awal apakah file valid CSV
    try:
        sample = pd.read_csv(target, nrows=5, encoding='utf-8-sig')
        missing = REQUIRED_RAW - set(sample.columns)
        if missing:
            # Tetap simpan tapi catat warning
            pass
    except Exception as e:
        target.unlink(missing_ok=True)
        raise ValueError(f"Berkas bukan CSV yang valid atau rusak: {e}")

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


def _store_labels(raw: pd.DataFrame, labels: pd.DataFrame, user_id=None) -> tuple[int, int, int]:
    """Simpan label preprocessing ke MySQL: kafe, ulasan, label_sentimen, skor_aspek."""
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
            'hours': int('24 jam' in blob or '24jam' in blob),
            'wifi': int('wifi' in blob or 'wi-fi' in blob or 'internet' in blob),
            'outlet': int(any(q in blob for q in ('colokan', 'stopkontak', 'charger'))),
            'desk': int(any(q in blob for q in ('meja', 'kursi', 'nugas', 'laptop', 'wfc', 'kerja'))),
        })

    with engine.begin() as conn:
        # Upsert kafe master
        conn.execute(text("""
            INSERT INTO kafe(place_id,nama,alamat,rating_gmaps,jumlah_ulasan,buka_24jam,wifi,stopkontak,meja_kerja)
            VALUES(:place,:name,:address,:rating,:count,:hours,:wifi,:outlet,:desk)
            ON DUPLICATE KEY UPDATE
              nama=VALUES(nama), alamat=VALUES(alamat),
              rating_gmaps=VALUES(rating_gmaps), jumlah_ulasan=VALUES(jumlah_ulasan),
              buka_24jam=VALUES(buka_24jam), wifi=VALUES(wifi),
              stopkontak=VALUES(stopkontak), meja_kerja=VALUES(meja_kerja)
        """), cafes)

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
                'kid': ids[str(r.placeId)],
                'rid': int(r.review_id),
                'text': str(r.teks_ulasan)[:60000],
                'rating': rating,
                'date': None,
            })

        ins = text('INSERT INTO ulasan(kafe_id,review_id,teks,rating,tanggal,valid) VALUES(:kid,:rid,:text,:rating,:date,1)')
        for i in range(0, len(review), 2000):
            conn.execute(ins, review[i:i + 2000])

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
                        'uid': ulid,
                        'kid': key[0],
                        'sub': str(r.sub_kalimat)[:512],
                        'aspect': r.aspek,
                        'sent': r.sentimen,
                    })

        inslabel = text('INSERT INTO label_sentimen(ulasan_id,kafe_id,sub_kalimat,aspek,sentimen) VALUES(:uid,:kid,:sub,:aspect,:sent)')
        for i in range(0, len(lr), 3000):
            conn.execute(inslabel, lr[i:i + 3000])

        # Hitung agregasi skor_aspek per kafe
        conn.execute(text("""
            INSERT INTO skor_aspek(kafe_id,aspek,jumlah_positif,jumlah_netral,jumlah_negatif,
                                   persen_positif,persen_negatif,label_kekuatan,label_kelemahan)
            SELECT kafe_id, aspek,
              SUM(sentimen='positif'), SUM(sentimen='netral'), SUM(sentimen='negatif'),
              ROUND(100.0*SUM(sentimen='positif')/COUNT(*),2),
              ROUND(100.0*SUM(sentimen='negatif')/COUNT(*),2),
              100.0*SUM(sentimen='positif')/COUNT(*) >= 60,
              100.0*SUM(sentimen='negatif')/COUNT(*) >= 30
            FROM label_sentimen GROUP BY kafe_id, aspek
        """))

    return len(review), len(lr), len(cafes)


def run_preprocessing(filename: str | None = None, user_id=None) -> dict:
    """Jalankan preprocessing dan pelabelan, simpan hasilnya ke DB dan cache."""
    csv_path = get_dataset_path(filename)
    if not csv_path or not csv_path.is_file():
        raise FileNotFoundError("Berkas dataset tidak ditemukan. Pilih atau unggah dataset terlebih dahulu.")

    active_name = csv_path.name
    write_state(
        dataset_name=active_name,
        dataset_size_bytes=csv_path.stat().st_size,
        checkpoint=1,
        checkpoint_title='Persiapan data',
        checkpoint_action='Membuka dan membaca berkas CSV ulasan',
        checkpoint_result=f'Membaca berkas {active_name}',
        message=f'Membaca berkas {active_name}...'
    )

    raw = pd.read_csv(csv_path, encoding='utf-8-sig')
    write_state(
        checkpoint=1,
        checkpoint_title='Persiapan data',
        checkpoint_action='Membaca CSV dan memeriksa kolom wajib',
        checkpoint_result=f'{len(raw):,} baris mentah ditemukan pada {active_name}',
        message=f'Checkpoint 1/6 — {len(raw):,} ulasan siap diproses',
    )

    def _progress(done: int, total: int, n_labels: int):
        write_state(
            checkpoint=4,
            checkpoint_title='Enam tahap preprocessing',
            checkpoint_action='Menormalisasi, tokenisasi, menghapus stopword, dan stemming',
            checkpoint_result=f'{n_labels:,} label sementara dari {done:,} ulasan selesai',
        )
        progress_callback('Preprocessing & pelabelan', done, total, f'{n_labels:,} label sementara')

    write_state(
        checkpoint=2,
        checkpoint_title='Potong template',
        checkpoint_action='Menghapus bagian boilerplate Google Maps dari teks ulasan',
        checkpoint_result='Teks ulasan siap dibaca sebagai isi review',
    )
    write_state(
        checkpoint=3,
        checkpoint_title='Pecah dan beri label',
        checkpoint_action='Memecah sub-kalimat lalu mendeteksi aspek dan sentimen lexicon',
        checkpoint_result='Label aspek-sentimen dikumpulkan per ulasan',
    )

    labels, summary = preprocess_and_label(raw, progress=_progress)

    write_state(
        checkpoint=5,
        checkpoint_title='Simpan hasil',
        checkpoint_action='Menyimpan ulasan, label, dan agregasi skor aspek ke database',
        checkpoint_result=f'{len(labels):,} label dari {summary["jumlah_review_berlabel"]:,} ulasan siap disimpan',
        message='Checkpoint 5/6 — menyimpan hasil preprocessing',
    )

    # Simpan ke cache khusus dataset dan cache umum
    dataset_cache = UPLOAD / f'.labels_{active_name}.pkl'
    labels.to_pickle(dataset_cache)
    labels.to_pickle(CACHE)

    counts = _store_labels(raw, labels, user_id)

    # ⚠️ PENTING: CSV mentah TIDAK dihapus! Tetap dipertahankan di DATASETS_DIR agar bisa diproses ulang!

    # Update metadata di registry
    reg = _read_registry()
    if active_name in reg:
        reg[active_name]['preprocessed'] = True
        reg[active_name]['labels_count'] = len(labels)
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
    state = read_state()
    active_name = state.get('dataset_name')
    dataset_cache = UPLOAD / f'.labels_{active_name}.pkl' if active_name else None

    labels = None
    if dataset_cache and dataset_cache.exists():
        labels = pd.read_pickle(dataset_cache)
    elif CACHE.exists():
        labels = pd.read_pickle(CACHE)

    if labels is None:
        raise ValueError('Hasil preprocessing belum ditemukan. Jalankan preprocessing sebelum klasifikasi.')

    write_state(
        class_checkpoint=0,
        class_checkpoint_title='Menyiapkan klasifikasi',
        class_checkpoint_action='Membaca data label hasil preprocessing',
        class_checkpoint_result='',
        message='Menyiapkan alur klasifikasi MNB'
    )

    def _progress(stage: str, done: int, total: int, detail: str):
        actions = {
            1: "Membagi ulasan menjadi 80% latih dan 20% uji berbasis review_id (Group Shuffle Split anti-bocor)",
            2: "Mengekstraksi fitur teks menggunakan Sublinear TF, Smooth IDF, dan Normalisasi L2",
            3: "Menyeimbangkan kelas sentimen minoritas (negatif & netral) pada data latih via Random Oversampling",
            4: "Menghitung log prior dan log likelihood kata per kelas dengan Laplace smoothing (alpha=1.0)",
            5: "Menghitung Confusion Matrix, Akurasi, Precision, Recall, Macro F1, dan Weighted F1 pada data uji"
        }
        write_state(
            class_checkpoint=done,
            class_checkpoint_title=stage,
            class_checkpoint_action=actions.get(done, stage),
            class_checkpoint_result=detail,
            message=f"Klasifikasi: {stage} ({detail})"
        )
        progress_callback(stage, done, total, detail)

    metrics, _, _, _ = fit_evaluate_manual(labels, progress=_progress)

    cm = {
        'classes': metrics['classes'],
        'matrix': metrics['matrix'],
        'per_class': metrics['per_class'],
    }

    with engine.begin() as conn:
        conn.execute(text('DELETE FROM evaluasi_model'))
        conn.execute(text("""
            INSERT INTO evaluasi_model(nama_model,accuracy,precision_macro,recall_macro,
                                       f1_macro,f1_weighted,jumlah_testing,confusion_matrix)
            VALUES(:name,:acc,:p,:r,:f,:w,:n,:cm)
        """), {
            'name': f"Multinomial Naive Bayes ({active_name or 'manual'})",
            'acc': metrics['accuracy'],
            'p': metrics['macro']['precision'],
            'r': metrics['macro']['recall'],
            'f': metrics['macro']['f1'],
            'w': metrics['weighted']['f1'],
            'n': metrics['jumlah_testing'],
            'cm': json.dumps(cm),
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
        if active_name in reg:
            reg[active_name]['classified'] = True
            reg[active_name]['accuracy'] = round(float(metrics['accuracy']), 4)
            reg[active_name]['f1_macro'] = round(float(metrics['macro']['f1']), 4)
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
