"""
Koneksi database Nafinder — konfigurasi via environment variable.

Kenapa pakai env var (bukan kredensial tertulis di kode):
  - Deploy tinggal ganti 1 tempat, kode tidak disentuh
  - Kredensial tidak bocor ke GitHub

Cara pakai:
  Local  : export DB_HOST=localhost DB_NAME=nafinder DB_USER=root DB_PASS=
  Deploy : set environment variable di panel server (Railway/Render/VPS)
           lalu sesuaikan nama variabelnya di sini bila berbeda.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()


def get_database_url() -> str:
    """Bangun URL MySQL dari variabel lingkungan, dengan fallback aman."""
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    name = os.getenv("DB_NAME", "nafinder")
    user = os.getenv("DB_USER", "root")
    pw = os.getenv("DB_PASS", "")
    # MySQL8 default auth (caching_sha2) butuh driver PyMySQL
    auth = f"{user}:{pw}" if pw else user
    return f"mysql+pymysql://{auth}@{host}:{port}/{name}?charset=utf8mb4"


engine = create_engine(
    get_database_url(),
    pool_pre_ping=True,   # deteksi koneksi MySQL yang putus
    pool_recycle=3600,    # reset koneksi tiap jam, cegah "gone away"
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    """Dependency: koneksi per request, selalu ditutup setelah selesai."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    # Tes koneksi cepat: python db.py
    from sqlalchemy import text

    with engine.connect() as conn:
        versi = conn.execute(text("SELECT VERSION()")).scalar()
    print(f"Terhubung ke MySQL {versi}")
    print(f"URL: {get_database_url().split('@')[-1]}")
