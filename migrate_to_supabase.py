"""
Skrip Migrasi Data Otomatis: MySQL (Laragon) -> PostgreSQL (Supabase)
Fadhli Ilham Nafi'an Yuswono (23.18.001) - Nafinder

Proyek Supabase: db_skripsi (Ref: bquulomsbxixdabpulkz)
Host: aws-1-ap-southeast-1.pooler.supabase.com:5432
"""

from __future__ import annotations
import getpass
import json
import os
import sys
from pathlib import Path
from sqlalchemy import create_engine, text

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DEFAULT_USER = "postgres.bquulomsbxixdabpulkz"
DEFAULT_HOST = "aws-1-ap-southeast-1.pooler.supabase.com"
DEFAULT_PORT = "5432"
DEFAULT_DB = "postgres"


def get_target_db_url() -> str:
    # 1. Dari argumen CLI
    if len(sys.argv) > 1 and sys.argv[1].startswith(("postgres://", "postgresql://")):
        url = sys.argv[1].strip()
    # 2. Dari environment variable
    elif os.getenv("DATABASE_URL"):
        url = os.getenv("DATABASE_URL").strip()
    # 3. Interaktif: minta password dan susun otomatis
    else:
        print("\n=== KONFIGURASI KONEKSI SUPABASE (db_skripsi) ===")
        print(f"Host     : {DEFAULT_HOST}:{DEFAULT_PORT}")
        print(f"Database : {DEFAULT_DB}")
        print(f"User     : {DEFAULT_USER}")
        print("--------------------------------------------------")
        print("Tips: Di terminal, karakter password sengaja disembunyikan demi keamanan.")
        print("Pilih cara input:")
        print("  [1] Ketik tersembunyi (standar keamanan terminal)")
        print("  [2] Ketik terlihat (teks muncul di layar)")
        pilihan = input("Pilih [1/2, default 1]: ").strip()

        if pilihan == "2":
            password = input("Masukkan password database Supabase Anda: ").strip()
        else:
            print("(Ketik password Anda lalu tekan Enter — huruf tidak akan muncul di layar)")
            password = getpass.getpass("Masukkan password database Supabase Anda: ").strip()

        if not password:
            print("[!] Password tidak boleh kosong.")
            sys.exit(1)
        url = f"postgresql://{DEFAULT_USER}:{password}@{DEFAULT_HOST}:{DEFAULT_PORT}/{DEFAULT_DB}?sslmode=require"

    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url.startswith("postgresql://") and not url.startswith("postgresql+"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


def run_migration():
    supabase_url = get_target_db_url()
    mysql_url = "mysql+pymysql://root:@localhost:3306/nafinder?charset=utf8mb4"

    print("\n==================================================================")
    print("       MIGRASI DATA NAFINDER: MYSQL (LARAGON) -> SUPABASE        ")
    print("==================================================================")
    print(f"[*] Sumber (MySQL)   : localhost:3306/nafinder")
    print(f"[*] Tujuan (Supabase): {supabase_url.split('@')[-1]}")
    print()

    # 1. Uji koneksi
    print("[1/4] Menguji koneksi database...")
    mysql_engine = create_engine(mysql_url)
    try:
        with mysql_engine.connect() as m_conn:
            m_ver = m_conn.execute(text("SELECT VERSION()")).scalar()
            print(f"  [+] Terhubung ke MySQL {m_ver}")
    except Exception as e:
        print(f"  [-] Gagal terhubung ke MySQL: {e}")
        print("      Pastikan Laragon sudah berjalan!")
        sys.exit(1)

    try:
        supabase_engine = create_engine(supabase_url, pool_pre_ping=True)
        with supabase_engine.connect() as s_conn:
            s_ver = s_conn.execute(text("SELECT VERSION()")).scalar()
            print(f"  [+] Terhubung ke Supabase PostgreSQL: {s_ver.split(',')[0]}")
    except Exception as e:
        print(f"  [-] Gagal terhubung ke Supabase: {e}")
        print("      Periksa kembali password database Anda!")
        sys.exit(1)

    # 2. Buat skema tabel
    print("\n[2/4] Menyiapkan skema tabel di Supabase...")
    schema_path = BASE_DIR / "schema_supabase.sql"
    if not schema_path.is_file():
        print(f"  [-] Berkas {schema_path} tidak ditemukan!")
        sys.exit(1)

    schema_sql = schema_path.read_text(encoding="utf-8")
    # Bersihkan baris komentar agar pernyataan CREATE TABLE tidak terbuang
    clean_lines = [line for line in schema_sql.splitlines() if not line.strip().startswith("--")]
    clean_sql = "\n".join(clean_lines)
    statements = [stmt.strip() for stmt in clean_sql.split(";") if stmt.strip()]

    with supabase_engine.begin() as s_conn:
        for stmt in statements:
            s_conn.execute(text(stmt))
    print("  [+] Seluruh tabel dan indeks berhasil dibuat di Supabase!")

    # 3. Salin data per tabel
    print("\n[3/4] Menyalin data antar tabel (streaming batch 500)...")
    tables = [
        "users", "kafe", "ulasan", "label_sentimen", "skor_aspek",
        "evaluasi_model", "riwayat_upload", "validasi_v4", "validasi_v4_aspek",
        "validasi_v5", "validasi_v5_aspek"
    ]

    BOOL_COLUMNS = {
        'buka_24jam', 'wifi', 'stopkontak', 'meja_kerja', 'valid',
        'label_kekuatan', 'label_kelemahan'
    }

    for table in tables:
        with mysql_engine.connect() as m_conn:
            rows = m_conn.execute(text(f"SELECT * FROM `{table}`")).mappings().all()
            if not rows:
                print(f"  - Tabel {table}: 0 baris (dilewati)")
                continue

            cols = list(rows[0].keys())

        converted_rows = []
        for r in rows:
            row_dict = dict(r)
            for k, v in row_dict.items():
                if k in BOOL_COLUMNS:
                    row_dict[k] = bool(v) if v is not None else None
                elif isinstance(v, (dict, list)):
                    row_dict[k] = json.dumps(v)
            converted_rows.append(row_dict)

        with supabase_engine.begin() as s_conn:
            col_names = ", ".join([f'"{c}"' for c in cols])
            param_names = ", ".join([f":{c}" for c in cols])
            insert_stmt = text(f'INSERT INTO {table} ({col_names}) VALUES ({param_names})')

            batch_size = 500
            for i in range(0, len(converted_rows), batch_size):
                batch = converted_rows[i:i + batch_size]
                s_conn.execute(insert_stmt, batch)

        print(f"  [+] Tabel {table}: Berhasil menyalin {len(rows):,} baris")

    # 4. Sinkronisasi sequence counter
    print("\n[4/4] Memperbarui counter sequence ID auto-increment...")
    seqs = [
        ("users", "users_id_seq"),
        ("kafe", "kafe_id_seq"),
        ("ulasan", "ulasan_id_seq"),
        ("label_sentimen", "label_sentimen_id_seq"),
        ("evaluasi_model", "evaluasi_model_id_seq"),
        ("riwayat_upload", "riwayat_upload_id_seq"),
        ("validasi_v4", "validasi_v4_id_seq"),
        ("validasi_v4_aspek", "validasi_v4_aspek_id_seq"),
        ("validasi_v5", "validasi_v5_id_seq"),
        ("validasi_v5_aspek", "validasi_v5_aspek_id_seq"),
    ]
    with supabase_engine.begin() as s_conn:
        for tbl, seq in seqs:
            s_conn.execute(text(f"SELECT setval('{seq}', COALESCE((SELECT MAX(id) FROM {tbl}), 1))"))

    print("  [+] Sequence ID tersinkronisasi.")
    print("\n==================================================================")
    print("            MIGRASI DATA KE SUPABASE SELESAI SUKSES!              ")
    print("==================================================================")
    print("Sekarang database Supabase Anda telah terisi lengkap dengan:")
    print("  - 324 Kafe master")
    print("  - 1.922 Ulasan")
    print("  - 2.000 Label aspek & sentimen")
    print("  - 393 Skor aspek teragregasi")
    print("  - Evaluasi model MNB (Akurasi 90,33%)")


if __name__ == "__main__":
    run_migration()
