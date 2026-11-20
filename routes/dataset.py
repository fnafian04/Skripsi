"""Dataset workflow route: upload multi-dataset, aktivasi, preprocessing, klasifikasi, unduh, hapus."""
import io
from pathlib import Path
import pandas as pd
from flask import (
    Blueprint, render_template, session, redirect, url_for,
    request, flash, jsonify, send_file
)
from werkzeug.utils import secure_filename
from db import SessionLocal
from models import Kafe, LabelSentimen, Ulasan
from processing import dataset_workflow as wf
from processing.workflow_state import is_process_active, read_state, run_background

bp = Blueprint('dataset', __name__, url_prefix='/dataset')
PRIMARY_DATASET = 'dataset_semprog.csv'
FIXED_DATASET = 'dataset_Nov_preprocessed.csv'
PROTECTED_DATASETS = {'dataset_Nov_preprocessed.csv', 'dataset_semprog.csv'}


def login_guard():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return None


@bp.route('/', methods=['GET', 'POST'])
def index():
    guard = login_guard()
    if guard:
        return guard

    current_state = read_state()
    fixed_path = wf.get_dataset_path(FIXED_DATASET)
    primary_path = wf.get_dataset_path(PRIMARY_DATASET)
    if fixed_path is None:
        flash(f"Dataset final '{FIXED_DATASET}' tidak tersedia di penyimpanan aplikasi.", 'error')
    elif not is_process_active():
        active_name = current_state.get('dataset_name')
        if not active_name or wf.get_dataset_path(active_name) is None:
            fallback_name = PRIMARY_DATASET if primary_path is not None else FIXED_DATASET
            wf.activate_dataset(fallback_name)
            current_state = read_state()

    datasets = wf.list_datasets()
    order_priority = {PRIMARY_DATASET: 0, FIXED_DATASET: 1}
    datasets.sort(key=lambda item: (order_priority.get(item['name'], 2), item['name']))

    return render_template(
        'dataset/index.html',
        state=current_state,
        datasets=datasets,
        fixed_dataset=FIXED_DATASET,
        protected_datasets=PROTECTED_DATASETS,
    )


@bp.post('/unggah')
def upload():
    guard = login_guard()
    if guard:
        return guard
    if is_process_active():
        flash('Tunggu sampai proses yang sedang berjalan selesai sebelum mengunggah dataset.', 'info')
        return redirect(url_for('dataset.index'))

    uploaded = request.files.get('dataset_file')
    if uploaded is None or not uploaded.filename:
        flash('Pilih file CSV terlebih dahulu.', 'error')
        return redirect(url_for('dataset.index'))

    filename = secure_filename(uploaded.filename)
    if not filename.lower().endswith('.csv'):
        flash('File yang diunggah harus berformat CSV.', 'error')
        return redirect(url_for('dataset.index'))
    if filename.lower() == FIXED_DATASET.lower():
        flash('Nama tersebut milik dataset final. Gunakan nama lain agar dataset final tetap tersimpan.', 'error')
        return redirect(url_for('dataset.index'))
    if filename.lower() in {p.lower() for p in PROTECTED_DATASETS}:
        flash('Nama tersebut milik dataset penelitian. Gunakan nama lain agar dataset tetap tersimpan.', 'error')
        return redirect(url_for('dataset.index'))
    if wf.get_dataset_path(filename) is not None:
        flash(f"Dataset '{filename}' sudah ada. Ubah nama file lalu coba lagi.", 'error')
        return redirect(url_for('dataset.index'))

    try:
        wf.save_dataset(uploaded, original_name=filename, set_active=True)
    except (OSError, ValueError) as e:
        flash(f'Gagal menambahkan dataset: {e}', 'error')
        return redirect(url_for('dataset.index'))

    flash(f"Dataset '{filename}' berhasil ditambahkan dan dipilih untuk uji coba.", 'success')
    return redirect(url_for('dataset.index'))


@bp.post('/aktifkan/<path:filename>')
def activate(filename):
    guard = login_guard()
    if guard:
        return guard

    if is_process_active():
        flash('Tidak dapat mengganti dataset aktif saat proses sedang berjalan, tunggu sampai selesai', 'info')
        return redirect(url_for('dataset.index'))

    try:
        wf.activate_dataset(filename)
        flash(f"Dataset '{filename}' berhasil diaktifkan", 'success')
    except (FileNotFoundError, OSError, ValueError) as e:
        flash(f"Gagal mengaktifkan dataset: {e}", 'error')

    return redirect(url_for('dataset.index'))


@bp.post('/preprocessing')
def preprocessing():
    guard = login_guard()
    if guard:
        return guard

    filename = request.form.get('filename')
    active_name = read_state().get('dataset_name')
    if not active_name or not wf.dataset_aktif_path():
        flash('Belum ada dataset aktif. Pilih dataset yang tersedia terlebih dahulu.', 'error')
        return redirect(url_for('dataset.index'))
    if filename and filename != active_name:
        flash('Dataset pada formulir berbeda dengan dataset aktif. Aktifkan dataset yang ingin diproses terlebih dahulu.', 'error')
        return redirect(url_for('dataset.index'))

    if not run_background('preprocessing', wf.run_preprocessing, None, session.get('user_id')):
        flash('Masih ada proses berjalan, tunggu sampai selesai', 'info')

    return redirect(url_for('dataset.index'))


@bp.post('/klasifikasi')
def classification():
    guard = login_guard()
    if guard:
        return guard

    state = read_state()
    if not state.get('dataset_name') or not wf.dataset_aktif_path():
        flash('Belum ada dataset aktif. Pilih dataset yang tersedia terlebih dahulu.', 'error')
        return redirect(url_for('dataset.index'))
    if not state.get('preprocessed'):
        flash('Klasifikasi belum bisa dijalankan, selesaikan preprocessing terlebih dahulu', 'error')
        return redirect(url_for('dataset.index'))

    if not run_background('classification', wf.run_classification, session.get('user_id')):
        flash('Masih ada proses berjalan, tunggu sampai selesai', 'info')

    return redirect(url_for('dataset.index'))


@bp.post('/hapus')
@bp.post('/hapus/<path:filename>')
def delete(filename=None):
    guard = login_guard()
    if guard:
        return guard
    if is_process_active():
        flash('Tidak dapat menghapus dataset saat proses sedang berjalan.', 'info')
        return redirect(url_for('dataset.index'))
    if not filename or filename.lower() in {p.lower() for p in PROTECTED_DATASETS}:
        flash('Dataset utama penelitian dikunci dan tidak dapat dihapus.', 'error')
        return redirect(url_for('dataset.index'))
    try:
        wf.delete_dataset(filename)
        flash(f"Dataset '{filename}' berhasil dihapus.", 'success')
    except OSError as e:
        flash(f"Gagal menghapus dataset: {e}", 'error')
    return redirect(url_for('dataset.index'))


@bp.post('/reset-database')
def reset_database():
    guard = login_guard()
    if guard:
        return guard
    flash('Data hasil anotasi dikunci dan tidak dapat dibersihkan.', 'error')
    return redirect(url_for('dataset.index'))


@bp.post('/reset-status')
def reset_status():
    guard = login_guard()
    if guard:
        return guard

    from processing.workflow_state import unlock_pipeline
    unlock_pipeline()
    flash('Kunci proses pipeline berhasil direset, Anda dapat menjalankan alur kembali', 'success')
    return redirect(url_for('dataset.index'))


@bp.get('/status')
def status():
    guard = login_guard()
    if guard:
        return jsonify({'error': 'unauthorized'}), 401

    # Auto-heal status jika worker thread di memori server tidak aktif
    is_process_active()

    state = read_state()
    datasets = wf.list_datasets()
    return jsonify({
        'state': state,
        'datasets': datasets,
        # Backward compatibility fields
        **state
    })


@bp.get('/unduh')
@bp.get('/unduh/<path:filename>')
def download_preprocessed(filename=None):
    """Unduh hasil preprocessing dataset dalam format CSV."""
    guard = login_guard()
    if guard:
        return guard

    state = read_state()
    target_name = Path(filename).name if filename else state.get('dataset_name')
    if not target_name:
        flash('Belum ada dataset yang dipilih untuk diunduh.', 'error')
        return redirect(url_for('dataset.index'))

    stem = Path(target_name).stem
    download_name = f"{stem}_preprocessed.csv" if 'preprocessed' not in stem.lower() else f"{stem}.csv"

    # 1. Prioritas: baca hasil cache pickle (.labels_<nama>.pkl)
    cache_candidates = [
        wf.UPLOAD / f'.labels_{target_name}.pkl',
        wf.BASE / 'uploads_tmp' / f'.labels_{target_name}.pkl',
    ]
    if target_name == state.get('dataset_name'):
        cache_candidates.extend([wf.CACHE, wf.BASE / 'uploads_tmp' / '.labels_cache.pkl'])

    for cache_path in cache_candidates:
        if cache_path and cache_path.is_file():
            df = pd.read_pickle(cache_path)
            buf = io.BytesIO(df.to_csv(index=False).encode('utf-8-sig'))
            buf.seek(0)
            return send_file(
                buf,
                as_attachment=True,
                download_name=download_name,
                mimetype='text/csv; charset=utf-8',
            )

    # 2. Fallback: CSV sumber yang sudah memiliki kolom teks_preprocessed
    csv_path = wf.get_dataset_path(target_name)
    if csv_path and csv_path.is_file():
        sample = pd.read_csv(csv_path, nrows=1, encoding='utf-8-sig')
        if 'teks_preprocessed' in sample.columns:
            return send_file(
                csv_path,
                as_attachment=True,
                download_name=download_name,
                mimetype='text/csv; charset=utf-8',
            )

    # 3. Fallback: ekspor dari database bila dataset aktif sudah diproses
    if target_name == state.get('dataset_name') and state.get('preprocessed'):
        db = SessionLocal()
        try:
            rows = (
                db.query(
                    Ulasan.review_id,
                    Kafe.place_id.label('placeId'),
                    Kafe.nama.label('nama_kafe'),
                    LabelSentimen.sub_kalimat,
                    LabelSentimen.aspek,
                    LabelSentimen.sentimen,
                    Ulasan.teks_preprocessed,
                )
                .join(Kafe, Kafe.id == LabelSentimen.kafe_id)
                .join(Ulasan, Ulasan.id == LabelSentimen.ulasan_id)
                .all()
            )
            if rows:
                df = pd.DataFrame([dict(r._mapping) for r in rows])
                buf = io.BytesIO(df.to_csv(index=False).encode('utf-8-sig'))
                buf.seek(0)
                return send_file(
                    buf,
                    as_attachment=True,
                    download_name=download_name,
                    mimetype='text/csv; charset=utf-8',
                )
        finally:
            db.close()

    flash(f"Hasil preprocessing untuk '{target_name}' belum tersedia. Jalankan preprocessing terlebih dahulu.", 'error')
    return redirect(url_for('dataset.index'))
