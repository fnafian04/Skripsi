"""Stage-aware data workflow persisted locally as a small JSON state file."""
from __future__ import annotations
import json
import os
import threading
import time
import uuid
from pathlib import Path

_BASE = Path(__file__).resolve().parents[1]
_IS_SERVERLESS = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))

if _IS_SERVERLESS or not os.access(_BASE, os.W_OK):
    import tempfile
    _STATE_DIR = Path(tempfile.gettempdir()) / 'nafinder_uploads'
else:
    _STATE_DIR = _BASE / 'uploads_tmp'

try:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

_STATE_PATH = _STATE_DIR / '.pipeline_state.json'
_BUNDLED_STATE = _BASE / 'uploads_tmp' / '.pipeline_state.json'
_LOCK = threading.RLock()


def read_state():
    with _LOCK:
        state = None
        for path_to_try in (_STATE_PATH, _BUNDLED_STATE):
            if path_to_try.is_file():
                try:
                    state = json.loads(path_to_try.read_text(encoding='utf-8'))
                    break
                except (OSError, ValueError):
                    pass

        if not state:
            state = {
                'dataset_name': None, 'preprocessed': False, 'classified': False,
                'dataset_size_bytes': None,
                'status': 'belum_dimulai', 'stage': None, 'percent': 0,
                'message': 'Unggah dataset untuk memulai', 'elapsed_seconds': 0,
                'preprocessing_seconds': None, 'classification_seconds': None,
                'started_at': None, 'finished_at': None, 'error': None,
                'checkpoint': 0, 'checkpoint_title': '', 'checkpoint_action': '',
                'checkpoint_result': '',
                'class_checkpoint': 0, 'class_checkpoint_title': '',
                'class_checkpoint_action': '', 'class_checkpoint_result': ''
            }

        state.setdefault('preprocessing_seconds', None)
        state.setdefault('classification_seconds', None)

        # Sinkronisasi otomatis dataset_name jika kosong tapi ada file di repositori
        if not state.get('dataset_name'):
            datasets_dir = _STATE_PATH.parent / 'datasets'
            if datasets_dir.is_dir():
                csvs = list(datasets_dir.glob('*.csv'))
                if csvs:
                    chosen = next((c for c in csvs if 'final' in c.name), csvs[0])
                    state['dataset_name'] = chosen.name
                    state['dataset_size_bytes'] = chosen.stat().st_size
                    if state.get('status') == 'belum_dimulai':
                        state['status'] = 'siap'
                        state['message'] = f"Dataset '{chosen.name}' siap diproses"

        active_name = state.get('dataset_name')
        if active_name and state.get('status') != 'berjalan':
            fname = Path(active_name).name
            label_cache = _STATE_PATH.parent / f'.labels_{fname}.pkl'
            if label_cache.exists():
                state['preprocessed'] = True

        if state.get('dataset_size_bytes') is None and active_name:
            fname = Path(active_name).name
            candidates = [
                _STATE_PATH.parent / 'datasets' / fname,
                _STATE_PATH.parent / fname,
            ]
            for c in candidates:
                if c.is_file():
                    state['dataset_size_bytes'] = c.stat().st_size
                    break

        return state


def write_state(**updates):
    with _LOCK:
        state = read_state()
        state.update(updates)
        payload = json.dumps(state, ensure_ascii=False, indent=2)
        temp_file = _STATE_PATH.with_suffix('.tmp')

        # Retry loop khusus Windows saat file sedang dibaca thread/proses lain
        for attempt in range(5):
            try:
                temp_file.write_text(payload, encoding='utf-8')
                temp_file.replace(_STATE_PATH)
                break
            except (OSError, PermissionError):
                if attempt == 4:
                    try:
                        _STATE_PATH.write_text(payload, encoding='utf-8')
                    except Exception:
                        pass
                else:
                    time.sleep(0.04)
        return state


def reset_state(dataset_name=None, dataset_size_bytes=None):
    return write_state(
        dataset_name=dataset_name,
        dataset_size_bytes=dataset_size_bytes,
        preprocessed=False,
        classified=False,
        status='siap' if dataset_name else 'belum_dimulai',
        stage=None,
        percent=0,
        message='Dataset siap diproses' if dataset_name else 'Unggah dataset untuk memulai',
        elapsed_seconds=0,
        preprocessing_seconds=None,
        classification_seconds=None,
        started_at=None,
        finished_at=None,
        error=None,
        run_id=None,
        checkpoint=0,
        checkpoint_title='',
        checkpoint_action='',
        checkpoint_result='',
        class_checkpoint=0,
        class_checkpoint_title='',
        class_checkpoint_action='',
        class_checkpoint_result=''
    )


_ACTIVE_WORKER_THREAD: threading.Thread | None = None


def is_process_active() -> bool:
    """Kembalikan True bila worker preprocessing/klasifikasi sedang aktif."""
    global _ACTIVE_WORKER_THREAD
    state = read_state()
    if state.get('status') not in ('berjalan', 'menunggu'):
        return False

    # Jika state tercatat aktif tapi thread Python di memori sudah tidak aktif
    if _ACTIVE_WORKER_THREAD is None or not _ACTIVE_WORKER_THREAD.is_alive():
        write_state(
            status='gagal',
            message='Proses terhenti atau belum berjalan, silakan klik tombol untuk memulai kembali',
            error='Thread proses latar belakang tidak aktif di server',
            finished_at=time.time()
        )
        return False

    return True


def unlock_pipeline() -> dict:
    """Buka kunci paksa status jika terjadi hang atau pengguna ingin membatalkan."""
    global _ACTIVE_WORKER_THREAD
    _ACTIVE_WORKER_THREAD = None
    state = read_state()
    ds_name = state.get('dataset_name')
    return write_state(
        status='siap' if ds_name else 'belum_dimulai',
        message='Status alur siap dijalankan kembali',
        error=None,
        finished_at=time.time(),
        started_at=None,
        elapsed_seconds=0,
        preprocessing_seconds=None,
        classification_seconds=None,
        percent=0,
        stage=None,
        checkpoint=0,
        checkpoint_title='',
        checkpoint_action='',
        checkpoint_result='',
        class_checkpoint=0,
        class_checkpoint_title='',
        class_checkpoint_action='',
        class_checkpoint_result=''
    )


def run_background(stage, function, *args, **kwargs):
    global _ACTIVE_WORKER_THREAD
    with _LOCK:
        if is_process_active():
            return False
        run_id = str(uuid.uuid4())
        write_state(
            status='menunggu',
            stage=stage,
            percent=0,
            message='Menyiapkan proses...',
            started_at=time.time(),
            finished_at=None,
            error=None,
            run_id=run_id
        )

    def worker():
        start = time.time()
        try:
            write_state(
                status='berjalan',
                stage=stage,
                message='Memulai komputasi...',
                elapsed_seconds=0,
                **{f'{stage}_seconds': None}
            )
            function(*args, **kwargs)
            s = read_state()
            duration = round(time.time() - start, 1)
            write_state(
                status='selesai',
                percent=100,
                message='Tahap selesai',
                elapsed_seconds=duration,
                **{f'{stage}_seconds': duration},
                finished_at=time.time(),
                preprocessed=True if stage == 'preprocessing' else s.get('preprocessed', False),
                classified=True if stage == 'classification' else s.get('classified', False)
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            duration = round(time.time() - start, 1)
            write_state(
                status='gagal',
                message='Tahap gagal dijalankan',
                error=str(exc),
                elapsed_seconds=duration,
                **{f'{stage}_seconds': duration},
                finished_at=time.time()
            )

    t = threading.Thread(target=worker, daemon=True)
    _ACTIVE_WORKER_THREAD = t
    t.start()
    return True


def progress_callback(stage_name, current, total, detail=''):
    percent = round(100 * current / total) if total else 0
    start = read_state().get('started_at') or time.time()
    write_state(
        percent=percent,
        message=f'{stage_name}: {current:,}/{total:,}' + (f' — {detail}' if detail else ''),
        elapsed_seconds=round(time.time() - start, 1)
    )
