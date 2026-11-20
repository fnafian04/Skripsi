"""
Koneksi database Nafinder — konfigurasi via environment variable.
Mendukung PostgreSQL (Supabase / Vercel / Render / Cloud) dan MySQL (Laragon lokal).

Prioritas:
  1. Jika DATABASE_URL diatur:
     - Digunakan otomatis untuk koneksi Supabase / PostgreSQL.
     - Di lingkungan Vercel/Serverless, otomatis menggunakan NullPool agar koneksi
       tidak menggantung dan tidak melebihi kuota koneksi Supabase.
  2. Jika DATABASE_URL tidak diatur (fallback lokal):
     - Membaca konfigurasi MySQL Laragon (DB_HOST, DB_NAME, DB_USER, DB_PASS, DB_PORT).
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool

Base = declarative_base()


def get_database_url() -> str:
    """Bangun URL database dengan prioritas DATABASE_URL (Supabase/PostgreSQL)."""
    raw_url = os.getenv("DATABASE_URL")
    if raw_url and raw_url.strip():
        url = raw_url.strip()
        # Standarisasi skema postgres:// menjadi postgresql://
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        # Gunakan driver psycopg2 secara eksplisit jika belum ada driver khusus
        if url.startswith("postgresql://") and not url.startswith("postgresql+"):
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        return url

    # Fallback ke MySQL lokal (Laragon)
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    name = os.getenv("DB_NAME", "nafinder")
    user = os.getenv("DB_USER", "root")
    pw = os.getenv("DB_PASS", "")
    auth = f"{user}:{pw}" if pw else user
    return f"mysql+pymysql://{auth}@{host}:{port}/{name}?charset=utf8mb4"


def create_app_engine():
    db_url = get_database_url()
    is_postgres = db_url.startswith("postgresql")
    is_serverless = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))

    if is_serverless and is_postgres:
        # Lingkungan Serverless Vercel: gunakan pool kecil yang dapat dipakai ulang selama container hangat
        return create_engine(
            db_url,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=2,
            max_overflow=3,
        )
    elif is_serverless:
        return create_engine(
            db_url,
            poolclass=NullPool,
            pool_pre_ping=True,
        )
    elif is_postgres:
        return create_engine(
            db_url,
            pool_pre_ping=True,
            pool_recycle=1800,
            pool_size=5,
            max_overflow=5
        )
    else:
        # MySQL lokal (Laragon)
        return create_engine(
            db_url,
            pool_pre_ping=True,
            pool_recycle=3600
        )


engine = create_app_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    """Dependency: koneksi per request, selalu ditutup setelah selesai."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    from sqlalchemy import text
    with engine.connect() as conn:
        versi = conn.execute(text("SELECT VERSION()")).scalar()
    dialect = engine.dialect.name
    print(f"Terhubung ke database {dialect} ({versi})")
    print(f"URL: {get_database_url().split('@')[-1]}")
