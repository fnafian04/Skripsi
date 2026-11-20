# Panduan Menjalankan, Migrasi Supabase & Deploy Vercel — Nafinder

Aplikasi **Nafinder** kini mendukung dual-database:
1. **Lokal (MySQL via Laragon)**: Cocok untuk pengujian & demonstrasi offline saat seminar/sidang.
2. **Cloud (PostgreSQL via Supabase)**: Cocok untuk deployment online di **Vercel** via GitHub.

---

## 1. Menjalankan Aplikasi di Lokal (Laragon)

1. Nyalakan Laragon (**Start All**), pastikan MySQL aktif di port 3306.
2. Buka terminal/PowerShell di folder program:
   ```powershell
   cd "D:\Skripsi\Road to Semhas\program"
   python app.py
   ```
3. Buka browser: `http://localhost:5000` (Login: `nafii` / `123456`).

---

## 2. Migrasi Database ke Supabase (PostgreSQL)

Untuk menghubungkan web ke Supabase:
1. Buka [https://supabase.com](https://supabase.com), buat akun dan buat proyek baru (misal: `nafinder-db`).
2. Masuk ke **Project Settings** > **Database**, salin **Connection String** bagian **URI** (atau **Connection Pooling / Port 6543** untuk Serverless).
3. **PILIHAN A (Otomatis lewat Skrip)**:
   Jalankan perintah ini di PowerShell:
   ```powershell
   python migrate_to_supabase.py "postgresql://postgres.[REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres?sslmode=require"
   ```
   Skrip otomatis membuat semua tabel, indeks, menyalin 4.800+ baris data penelitian, dan mengatur counter sequence.

4. **PILIHAN B (Lewat Supabase SQL Editor)**:
   - Buka menu **SQL Editor** di dashboard Supabase.
   - Buka berkas `supabase_dump.sql` di folder program, salin seluruh isinya, tempel ke SQL Editor Supabase, lalu klik **Run**.

---

## 3. Unggah ke GitHub

Folder `program` telah diinisialisasi sebagai repositori Git branch `main`.
1. Buat repository baru di [GitHub](https://github.com/new) (misal: `nafinder`). Jangan centang README/gitignore (biarkan kosong).
2. Jalankan berkas:
   ```bat
   push_ke_github.bat
   ```
   Masukkan URL repository GitHub Anda (contoh: `https://github.com/username/nafinder.git`).
   Berkas akan otomatis diunggah ke GitHub.

---

## 4. Deploy ke Vercel

1. Buka [https://vercel.com](https://vercel.com) dan login dengan akun GitHub Anda.
2. Klik **Add New...** > **Project**, lalu pilih repositori `nafinder` yang baru diunggah.
3. Pada halaman konfigurasi project (**Configure Project**):
   - **Framework Preset**: Pilih `Other`.
   - **Root Directory**: `./` (default).
4. Buka bagian **Environment Variables**, tambahkan dua variabel penting:
   - `SECRET_KEY`: `nafinder-super-secret-key-2026` (atau string acak bebas)
   - `DATABASE_URL`: Masukkan URL koneksi Supabase Anda (pilih yang **Connection Pooling / Port 6543 / 5432** agar ramah serverless):
     `postgresql://postgres.[REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres?sslmode=require`
5. Klik **Deploy**.
6. Vercel akan otomatis meng-install dependensi dari `requirements.txt` dan menjalankan serverless function via `api/index.py`. Web Nafinder Anda langsung aktif dengan domain `.vercel.app`!

---

## 5. Struktur Berkas Kunci untuk Cloud & Vercel

- `api/index.py` — Entrypoint WSGI Serverless Function untuk Vercel.
- `vercel.json` — Konfigurasi rewrite routing Vercel ke `/api/index`.
- `db.py` — Engine database cerdas: otomatis memakai Supabase jika `DATABASE_URL` terdeteksi, dan fallback ke MySQL Laragon jika tidak ada.
- `models/tables.py` — Skema ORM multi-dialect (PostgreSQL & MySQL).
- `schema_supabase.sql` — DDL skema murni PostgreSQL untuk Supabase.
- `supabase_dump.sql` — DDL + seluruh data hasil penelitian siap eksekusi di Supabase.
- `migrate_to_supabase.py` — Skrip sinkronisasi 1-klik dari MySQL ke Supabase.
- `processing/assets/` — Model MNB (`model_mnb.pkl`) & vectorizer (`vectorizer.pkl`) pre-trained siap pakai di Vercel tanpa delay cold start.
