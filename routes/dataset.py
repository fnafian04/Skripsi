"""Dataset workflow route: upload multi-dataset, aktivasi, preprocessing, klasifikasi, hapus."""
from flask import (
    Blueprint, render_template, session, redirect, url_for,
    request, flash, jsonify
)
from werkzeug.utils import secure_filename
from pathlib import Path

from processing import dataset_workflow as wf
from processing.workflow_state import is_process_active, read_state, run_background

bp = Blueprint('dataset', __name__, url_prefix='/dataset')


def login_guard():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return None


@bp.route('/', methods=['GET', 'POST'])
def index():
    guard = login_guard()
    if guard:
        return guard

    if request.method == 'POST':
        if is_process_active():
            flash('Tidak dapat mengunggah saat proses sedang berjalan, tunggu sampai selesai', 'info')
            return redirect(url_for('dataset.index'))

        files = request.files.getlist('dataset')
        if not files or all(not f.filename for f in files):
            flash('Pilih setidaknya satu file CSV ulasan mentah', 'error')
            return redirect(url_for('dataset.index'))

        uploaded_count = 0
        last_uploaded_name = None

        for file in files:
            if not file or not file.filename:
                continue
            if not file.filename.lower().endswith('.csv'):
                flash(f"Berkas '{file.filename}' dilewati karena bukan format .csv", 'error')
                continue

            clean_name = secure_filename(file.filename) or 'dataset.csv'
            try:
                wf.save_dataset(file, original_name=clean_name, set_active=(uploaded_count == 0))
                uploaded_count += 1
                last_uploaded_name = clean_name
            except Exception as e:
                flash(f"Gagal menyimpan '{clean_name}': {e}", 'error')

        if uploaded_count > 0:
            flash(f"Berhasil mengunggah {uploaded_count} dataset ke repositori", 'success')
            if last_uploaded_name:
                flash(f"Dataset '{last_uploaded_name}' telah diaktifkan, Anda dapat langsung menjalankan preprocessing", 'info')

        return redirect(url_for('dataset.index'))

    datasets = wf.list_datasets()
    current_state = read_state()

    # Jika state belum memiliki dataset_name tapi ada dataset di direktori, aktifkan yang pertama
    if not current_state.get('dataset_name') and datasets:
        wf.activate_dataset(datasets[0]['name'])
        current_state = read_state()
        datasets = wf.list_datasets()

    return render_template(
        'dataset/index.html',
        state=current_state,
        datasets=datasets
    )


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
    except Exception as e:
        flash(f"Gagal mengaktifkan dataset: {e}", 'error')

    return redirect(url_for('dataset.index'))


@bp.post('/preprocessing')
def preprocessing():
    guard = login_guard()
    if guard:
        return guard

    filename = request.form.get('filename')
    if filename:
        try:
            wf.activate_dataset(filename)
        except Exception as e:
            flash(f"Gagal mengaktifkan dataset '{filename}': {e}", 'error')
            return redirect(url_for('dataset.index'))

    if not wf.dataset_aktif_path():
        flash('Pilih atau unggah dataset terlebih dahulu', 'error')
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
        flash('Tidak dapat menghapus dataset saat proses sedang berjalan, tunggu sampai selesai', 'info')
        return redirect(url_for('dataset.index'))

    target = filename or request.form.get('filename') or read_state().get('dataset_name')
    if not target:
        flash('Tidak ada dataset yang dipilih untuk dihapus', 'info')
        return redirect(url_for('dataset.index'))

    wf.delete_dataset(target)
    flash(f"Dataset '{target}' berhasil dihapus dari repositori", 'success')
    return redirect(url_for('dataset.index'))


@bp.post('/reset-database')
def reset_database():
    guard = login_guard()
    if guard:
        return guard

    if is_process_active():
        flash('Tidak dapat mereset saat proses sedang berjalan', 'info')
        return redirect(url_for('dataset.index'))

    wf.clear_pipeline_database()
    flash('Hasil ulasan dan evaluasi di database telah dibersihkan, berkas dataset mentah tetap tersimpan', 'success')
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

