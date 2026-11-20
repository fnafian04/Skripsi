"""Blueprint hasil klasifikasi — metrik model & confusion matrix."""

from flask import Blueprint, render_template, session, redirect, url_for, request, jsonify
from db import SessionLocal
from models import EvaluasiModel
from processing import dataset_workflow as wf

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


@bp.post('/uji-ulasan')
def test_review():
    guard = _login_required()
    if guard:
        return jsonify({'error': 'Sesi login tidak valid.'}), 401

    payload = request.get_json(silent=True) or {}
    text = str(payload.get('text', '')).strip()
    if not text:
        return jsonify({'error': 'Tulis ulasan terlebih dahulu.'}), 400
    if len(text) > 2000:
        return jsonify({'error': 'Ulasan maksimal 2.000 karakter.'}), 400

    try:
        predictions = wf.predict_review(text)
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 409
    except (TypeError, ValueError) as exc:
        return jsonify({'error': f'Ulasan tidak dapat diproses: {exc}'}), 422

    if not predictions:
        return jsonify({'error': 'Aspek belum terdeteksi. Tambahkan kata seperti makanan, harga, tempat, pelayanan, wifi, atau kebersihan.'}), 422
    return jsonify({'predictions': predictions})
