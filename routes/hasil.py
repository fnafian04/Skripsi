"""Blueprint hasil klasifikasi — metrik model & confusion matrix."""

from flask import Blueprint, render_template, session, redirect, url_for
from db import SessionLocal
from models import EvaluasiModel

bp = Blueprint('hasil', __name__, url_prefix='/hasil')


def _login_required():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return None


@bp.route('/')
def index():
    guard = _login_required()
    if guard:
        return guard

    db = SessionLocal()
    try:
        import json
        evaluasi = db.query(EvaluasiModel).order_by(
            EvaluasiModel.dilatih_pada.desc()
        ).first()

        cm = None
        if evaluasi and evaluasi.confusion_matrix:
            cm = evaluasi.confusion_matrix
            if isinstance(cm, str):
                try:
                    cm = json.loads(cm)
                except (TypeError, ValueError):
                    cm = None

        return render_template('hasil/index.html', evaluasi=evaluasi, cm=cm)
    finally:
        db.close()
