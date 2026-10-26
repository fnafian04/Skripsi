"""Validasi sampel label v4 dari CSV eksperimen, terpisah dari baseline."""
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from sqlalchemy import func

from db import SessionLocal, engine
from models import ValidasiV4, ValidasiV4Aspek

bp = Blueprint('validasi_v4', __name__, url_prefix='/validasi-v4')
PER_PAGE = 20
STATUSES = {'benar', 'salah', 'ragu'}
ASPEK = {'makanan_minuman', 'pelayanan', 'harga', 'suasana', 'fasilitas_wfc', 'kebersihan'}
SENTIMEN = {'positif', 'netral', 'negatif'}
CSV_PATH = Path(__file__).resolve().parents[1] / 'uploads_tmp' / 'labels_experiment_v4.csv'


def _guard():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return None


def _tables():
    ValidasiV4.__table__.create(bind=engine, checkfirst=True)
    ValidasiV4Aspek.__table__.create(bind=engine, checkfirst=True)


@bp.get('/')
def index():
    if (guard := _guard()):
        return guard
    _tables()
    db = SessionLocal()
    try:
        status = request.args.get('status', '')
        page = max(request.args.get('page', 1, type=int), 1)
        query = db.query(ValidasiV4)
        if status in STATUSES:
            query = query.filter(ValidasiV4.status == status)
        elif status == 'belum':
            query = query.filter(ValidasiV4.status.is_(None))
        total = query.count()
        pages = max((total + PER_PAGE - 1) // PER_PAGE, 1)
        page = min(page, pages)
        rows = query.order_by(ValidasiV4.urutan).offset((page - 1) * PER_PAGE).limit(PER_PAGE).all()
        total_sample = db.query(func.count(ValidasiV4.id)).scalar() or 0
        checked = db.query(func.count(ValidasiV4.id)).filter(ValidasiV4.status.isnot(None)).scalar() or 0
        counts = {s: db.query(func.count(ValidasiV4.id)).filter(ValidasiV4.status == s).scalar() or 0 for s in STATUSES}
        corrections = {}
        ids = [row.id for row in rows]
        if ids:
            for item in db.query(ValidasiV4Aspek).filter(ValidasiV4Aspek.validasi_id.in_(ids)).all():
                corrections.setdefault(item.validasi_id, set()).add(item.aspek)
        return render_template('validasi/v4.html', rows=rows, page=page, pages=pages, total=total,
            total_sample=total_sample, checked=checked, counts=counts, status_filter=status,
            aspek=sorted(ASPEK), sentimen=sorted(SENTIMEN), corrections=corrections)
    finally:
        db.close()


@bp.post('/mulai')
def start():
    if (guard := _guard()):
        return guard
    _tables()
    db = SessionLocal()
    try:
        if db.query(func.count(ValidasiV4.id)).scalar():
            flash('Batch v4 yang sudah ada dipertahankan', 'info')
            return redirect(url_for('validasi_v4.index'))
        if not CSV_PATH.exists():
            flash('File label v4 belum tersedia. Jalankan eksperimen v4 terlebih dahulu', 'error')
            return redirect(url_for('validasi_v4.index'))
        data = pd.read_csv(CSV_PATH)
        sample = data.drop_duplicates(['sub_kalimat']).sample(
            n=min(100, len(data)), random_state=20261025
        )
        db.bulk_save_objects([
            ValidasiV4(urutan=index, sub_kalimat=str(row.sub_kalimat)[:2048],
                teks_ulasan=str(row.asal_sub_kalimat), rating=int(row.rating_ulasan) if pd.notna(row.rating_ulasan) else None,
                aspek=str(row.aspek), sentimen=str(row.sentimen))
            for index, row in enumerate(sample.itertuples(index=False), 1)
        ])
        db.commit()
        flash(f'{len(sample)} label v4 siap divalidasi', 'success')
        return redirect(url_for('validasi_v4.index'))
    finally:
        db.close()


@bp.post('/simpan')
def save():
    if (guard := _guard()):
        return guard
    _tables()
    db = SessionLocal()
    try:
        for raw_id in request.form.getlist('validation_id'):
            item = db.query(ValidasiV4).filter(ValidasiV4.id == int(raw_id)).one_or_none()
            if item is None:
                continue
            status = request.form.get(f'status_{item.id}', '')
            item.status = status if status in STATUSES else None
            sentiment = request.form.get(f'sentimen_{item.id}', '')
            item.sentimen_koreksi = sentiment if sentiment in SENTIMEN else None
            item.catatan = request.form.get(f'catatan_{item.id}', '').strip()[:500] or None
            item.diperiksa_pada = datetime.utcnow()
            db.query(ValidasiV4Aspek).filter(ValidasiV4Aspek.validasi_id == item.id).delete(synchronize_session=False)
            selected = list(dict.fromkeys(value for value in request.form.getlist(f'aspek_{item.id}[]') if value in ASPEK))
            for aspect in selected:
                db.add(ValidasiV4Aspek(validasi_id=item.id, aspek=aspect))
        db.commit()
        flash('Checklist v4 tersimpan', 'success')
        return redirect(url_for('validasi_v4.index', page=request.form.get('page', 1), status=request.form.get('status_filter', '')))
    finally:
        db.close()
