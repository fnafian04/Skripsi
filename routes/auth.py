"""
Blueprint autentikasi — login & register (kerangka).

Belum ada pengecekan password sungguhan: fungsi hash & query DB sudah
ditulis, tinggal dirangkai saat bagian login dikerjakan beneran.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from db import SessionLocal
from models import User

bp = Blueprint('auth', __name__, url_prefix='')


@bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == username).first()
            # TODO: ganti ke check_password_hash saat data user sudah ada
            if user and check_password_hash(user.password_hash, password):
                session['user_id'] = user.id
                session['username'] = user.username
                return redirect(url_for('dashboard.index'))
            flash('Username atau password salah.', 'error')
        finally:
            db.close()

    return render_template('auth/login.html')


@bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if not username or not email or not password:
            flash('Semua kolom wajib diisi.', 'error')
            return render_template('auth/register.html')

        db = SessionLocal()
        try:
            existing = db.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            if existing:
                flash('Username atau email sudah terdaftar.', 'error')
                return render_template('auth/register.html')

            user = User(
                username=username,
                email=email,
                password_hash=generate_password_hash(password, method='pbkdf2:sha256')
            )
            db.add(user)
            db.commit()
            flash('Registrasi berhasil, silakan login.', 'success')
            return redirect(url_for('auth.login'))
        finally:
            db.close()

    return render_template('auth/register.html')


@bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('auth.login'))
