"""Dashboard ringkasan hasil pipeline Nafinder."""
from flask import Blueprint, render_template, session, redirect, url_for
from db import SessionLocal
from models import EvaluasiModel, Kafe, LabelSentimen, Ulasan
from sqlalchemy import func, case

bp = Blueprint('dashboard', __name__, url_prefix='/dashboard')


def _login_required():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return None


@bp.route('/')
def index():
    guard = _login_required()
    if guard: return guard
    db = SessionLocal()
    try:
        # Kafe master dipertahankan saat dataset dihapus. Dashboard menampilkan
        # kafe yang benar-benar masih memiliki ulasan pada dataset aktif.
        jumlah_kafe = db.query(func.count(func.distinct(Ulasan.kafe_id))).scalar() or 0
        jumlah_ulasan = db.query(func.count(Ulasan.id)).scalar() or 0
        jumlah_label = db.query(func.count(LabelSentimen.id)).scalar() or 0
        evaluasi = db.query(EvaluasiModel).order_by(EvaluasiModel.dilatih_pada.desc()).first()
        aspek_stats = db.query(
            LabelSentimen.aspek,
            func.count(LabelSentimen.id).label('total'),
            func.sum(case((LabelSentimen.sentimen == 'positif',1),else_=0)).label('positif'),
            func.sum(case((LabelSentimen.sentimen == 'netral',1),else_=0)).label('netral'),
            func.sum(case((LabelSentimen.sentimen == 'negatif',1),else_=0)).label('negatif'),
        ).group_by(LabelSentimen.aspek).order_by(func.count(LabelSentimen.id).desc()).all()
        return render_template('dashboard.html', jumlah_kafe=jumlah_kafe,
            jumlah_ulasan=jumlah_ulasan, jumlah_label=jumlah_label,
            evaluasi=evaluasi, aspek_stats=aspek_stats)
    finally:
        db.close()


@bp.route('/bantuan')
def bantuan():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return render_template('dashboard/bantuan.html')
