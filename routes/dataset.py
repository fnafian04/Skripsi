"""Dataset workflow route: upload multi-dataset, aktivasi, preprocessing, klasifikasi, hapus."""
from flask import (
    Blueprint, render_template, session, redirect, url_for,
    request, flash, jsonify
)
from werkzeug.utils import secure_filename
from processing import dataset_workflow as wf
from processing.workflow_state import is_process_active, read_state, run_background

bp = Blueprint('dataset', __name__, url_prefix='/dataset')
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
    if fixed_path is None:
        flash(f"Dataset final '{FIXED_DATASET}' tidak tersedia di penyimpanan aplikasi.", 'error')
    elif not is_process_active():
        active_name = current_state.get('dataset_name')
        if not active_name or wf.get_dataset_path(active_name) is None:
            wf.activate_dataset(FIXED_DATASET)
            current_state = read_state()

    datasets = wf.list_datasets()
    datasets.sort(key=lambda item: item['name'] != FIXED_DATASET)

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
