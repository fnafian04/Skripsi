"""Validasi manual sampel label tanpa mengubah label hasil pipeline."""

from datetime import datetime

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from sqlalchemy import and_, func

from db import Base, SessionLocal, engine
from models import LabelSentimen, Ulasan, ValidasiLabel, ValidasiLabelAspek

bp = Blueprint('validasi', __name__, url_prefix='/validasi')

PER_PAGE = 20
ALLOWED_STATUS = {'benar', 'salah', 'ragu'}
STATUS_FILTERS = ALLOWED_STATUS | {'belum'}
ALLOWED_ASPEK = {
    'makanan_minuman', 'pelayanan', 'harga', 'suasana',
    'fasilitas_wfc', 'kebersihan',
}
ALLOWED_SENTIMEN = {'positif', 'netral', 'negatif'}


def _login_required():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return None


def _ensure_table():
    ValidasiLabel.__table__.create(bind=engine, checkfirst=True)
    ValidasiLabelAspek.__table__.create(bind=engine, checkfirst=True)


@bp.get('/')
def index():
    guard = _login_required()
    if guard:
        return guard
    _ensure_table()
    db = SessionLocal()
    try:
        page = max(request.args.get('page', 1, type=int), 1)
        status_filter = request.args.get('status', '')
        query = db.query(ValidasiLabel, LabelSentimen, Ulasan).join(
            LabelSentimen, ValidasiLabel.label_id == LabelSentimen.id
        ).join(Ulasan, LabelSentimen.ulasan_id == Ulasan.id)
        if status_filter in ALLOWED_STATUS:
            query = query.filter(ValidasiLabel.status == status_filter)
        elif status_filter == 'belum':
            query = query.filter(ValidasiLabel.status.is_(None))
        total = query.count()
        pages = max((total + PER_PAGE - 1) // PER_PAGE, 1)
        page = min(page, pages)
        rows = query.order_by(ValidasiLabel.urutan).offset(
            (page - 1) * PER_PAGE
        ).limit(PER_PAGE).all()
        label_groups = {}
        correction_aspects = {}
        for _, label, _ in rows:
            key = (label.ulasan_id, label.sub_kalimat)
            label_groups[key] = db.query(LabelSentimen).filter(
                and_(
                    LabelSentimen.ulasan_id == label.ulasan_id,
                    LabelSentimen.sub_kalimat == label.sub_kalimat,
                )
            ).order_by(LabelSentimen.id).all()
        validation_ids = [validation.id for validation, _, _ in rows]
        if validation_ids:
            correction_rows = db.query(ValidasiLabelAspek).filter(
                ValidasiLabelAspek.validasi_id.in_(validation_ids)
            ).all()
            correction_aspects = {}
            for correction in correction_rows:
                correction_aspects.setdefault(correction.validasi_id, set()).add(
                    correction.aspek
                )
        for validation, _, _ in rows:
            if validation.aspek_koreksi:
                correction_aspects.setdefault(validation.id, set()).add(
                    validation.aspek_koreksi
                )
        sample_total = db.query(func.count(ValidasiLabel.id)).scalar() or 0
        checked = db.query(func.count(ValidasiLabel.id)).filter(
            ValidasiLabel.status.isnot(None)
        ).scalar() or 0
        counts = {
            status: db.query(func.count(ValidasiLabel.id)).filter(
                ValidasiLabel.status == status
            ).scalar() or 0
            for status in ALLOWED_STATUS
        }
        return render_template(
            'validasi/index.html', rows=rows, page=page, pages=pages,
            total=total, sample_total=sample_total, checked=checked,
            counts=counts, status_filter=status_filter,
            aspek=sorted(ALLOWED_ASPEK), sentimen=sorted(ALLOWED_SENTIMEN),
            label_groups=label_groups, correction_aspects=correction_aspects,
            status_filters=sorted(STATUS_FILTERS),
        )
    finally:
        db.close()


@bp.post('/mulai')
def start():
    guard = _login_required()
    if guard:
        return guard
    _ensure_table()
    size = request.form.get('size', '100', type=int)
    size = min(max(size, 20), 5000)
    db = SessionLocal()
    try:
        existing = db.query(func.count(ValidasiLabel.id)).scalar() or 0
        if existing:
            flash('Sampel validasi yang sedang berjalan dipertahankan agar checklist tidak hilang', 'info')
            return redirect(url_for('validasi.index'))
        ids = [
            row[0] for row in db.query(LabelSentimen.id)
            .order_by(func.rand()).limit(size).all()
        ]
        if not ids:
            flash('Belum ada label. Jalankan preprocessing terlebih dahulu', 'error')
            return redirect(url_for('validasi.index'))
        db.bulk_save_objects([
            ValidasiLabel(label_id=label_id, urutan=index)
            for index, label_id in enumerate(ids, 1)
        ])
        db.commit()
        flash(f'{len(ids):,} label dipilih untuk validasi manual', 'success')
        return redirect(url_for('validasi.index'))
    finally:
        db.close()


@bp.post('/simpan')
def save():
    guard = _login_required()
    if guard:
        return guard
    _ensure_table()
    db = SessionLocal()
    try:
        ids = request.form.getlist('validation_id')
        for raw_id in ids:
            validation_id = int(raw_id)
            row = db.query(ValidasiLabel).filter(
                ValidasiLabel.id == validation_id
            ).one_or_none()
            if row is None:
                continue
            status = request.form.get(f'status_{validation_id}', '').strip()
            row.status = status if status in ALLOWED_STATUS else None
            sentiment = request.form.get(f'sentimen_{validation_id}', '').strip()
            db.query(ValidasiLabelAspek).filter(
                ValidasiLabelAspek.validasi_id == validation_id
            ).delete(synchronize_session=False)
            selected_aspects = [
                value for value in request.form.getlist(f'aspek_{validation_id}[]')
                if value in ALLOWED_ASPEK
            ]
            selected_aspects = list(dict.fromkeys(selected_aspects))
            row.aspek_koreksi = selected_aspects[0] if selected_aspects else None
            for selected_aspect in selected_aspects:
                db.add(ValidasiLabelAspek(
                    validasi_id=validation_id,
                    aspek=selected_aspect,
                ))
            row.sentimen_koreksi = sentiment if sentiment in ALLOWED_SENTIMEN else None
            row.catatan = request.form.get(f'catatan_{validation_id}', '').strip()[:500] or None
            row.user_id = session['user_id']
            row.diperiksa_pada = datetime.utcnow()
        db.commit()
        flash('Checklist validasi tersimpan', 'success')
        return redirect(url_for('validasi.index', page=request.form.get('page', 1)))
    finally:
        db.close()
