"""Dashboard ringkasan hasil pipeline Nafinder."""
import threading
import time
from flask import Blueprint, render_template, session, redirect, url_for
from db import SessionLocal
from models import EvaluasiModel, Kafe, LabelSentimen, Ulasan
from processing.workflow_state import read_state
from sqlalchemy import func, case

bp = Blueprint('dashboard', __name__, url_prefix='/dashboard')

_DASH_TTL = 60.0
_DASH_LOCK = threading.Lock()
_DASH_CACHE: dict = {'key': None, 'ts': 0.0, 'data': None}


def _login_required():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return None


@bp.route('/')
def index():
    guard = _login_required()
    if guard:
        return guard

    state = read_state()
    state_key = (state.get('dataset_name'), state.get('run_id'), state.get('finished_at'))
    now = time.monotonic()
    with _DASH_LOCK:
        cached = _DASH_CACHE.get('data')
        if cached and _DASH_CACHE.get('key') == state_key and (now - _DASH_CACHE.get('ts', 0.0)) < _DASH_TTL:
            return render_template('dashboard.html', **cached)

    db = SessionLocal()
    try:
        jumlah_kafe = db.query(func.count(func.distinct(Ulasan.kafe_id))).scalar() or 0
        jumlah_ulasan = db.query(func.count(Ulasan.id)).scalar() or 0
        ev = db.query(EvaluasiModel).order_by(EvaluasiModel.dilatih_pada.desc()).first()
        evaluasi = None
        if ev:
            evaluasi = {
                'accuracy': float(ev.accuracy),
                'precision_macro': float(ev.precision_macro),
                'recall_macro': float(ev.recall_macro),
                'f1_macro': float(ev.f1_macro),
                'f1_weighted': float(ev.f1_weighted),
                'jumlah_testing': int(ev.jumlah_testing),
            }

        raw_stats = db.query(
            LabelSentimen.aspek,
            func.count(LabelSentimen.id).label('total'),
            func.sum(case((LabelSentimen.sentimen == 'positif', 1), else_=0)).label('positif'),
            func.sum(case((LabelSentimen.sentimen == 'netral', 1), else_=0)).label('netral'),
            func.sum(case((LabelSentimen.sentimen == 'negatif', 1), else_=0)).label('negatif'),
        ).group_by(LabelSentimen.aspek).order_by(func.count(LabelSentimen.id).desc()).all()

        aspek_stats = [
            {
                'aspek': row.aspek,
                'total': int(row.total or 0),
                'positif': int(row.positif or 0),
                'netral': int(row.netral or 0),
                'negatif': int(row.negatif or 0),
            }
            for row in raw_stats
        ]
        jumlah_label = sum(a['total'] for a in aspek_stats)
        payload = {
            'jumlah_kafe': int(jumlah_kafe),
            'jumlah_ulasan': int(jumlah_ulasan),
            'jumlah_label': int(jumlah_label),
            'evaluasi': evaluasi,
            'aspek_stats': aspek_stats,
        }
        with _DASH_LOCK:
            _DASH_CACHE.update({'key': state_key, 'ts': time.monotonic(), 'data': payload})
        return render_template('dashboard.html', **payload)
    finally:
        db.close()


@bp.route('/bantuan')
def bantuan():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return render_template('dashboard/bantuan.html')
