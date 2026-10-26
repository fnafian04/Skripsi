# Nafinder — panduan menjalankan aplikasi dan pipeline

## Pertama kali

1. Nyalakan Laragon dengan **Start All**. Pastikan MySQL/MariaDB aktif.
2. Buka PowerShell di folder `nafinder_app`.
3. Jika database belum dibuat, jalankan `schema.sql` sekali dari Terminal Laragon:

```powershell
Get-Content .\schema.sql | mysql -u root
```

Jika perintah `mysql` tidak dikenali, buka Terminal Laragon (yang menambahkan MySQL ke PATH) lalu jalankan perintah yang sama dari folder aplikasi.

4. Install dependensi satu kali:

```powershell
python -m pip install -r requirements.txt
```

Seluruh perhitungan metode, termasuk split anti-bocor, TF-IDF, oversampling, MNB, dan evaluasi, ditulis manual di `processing/`. Skrip `processing/validate_methods.py` saja memakai sklearn sebagai pembanding audit; aplikasi tidak memerlukannya.

Folder `uploads_tmp` adalah penyimpanan runtime lokal. Isinya dapat berupa
state progres, cache label sementara, staging upload sebelum preprocessing,
dan fixture pengujian. Folder `notebooks` hanya untuk notebook pengujian
metode; hasil notebook tidak digunakan otomatis oleh aplikasi.

## Setiap kali menjalankan

```powershell
cd "C:\path\ke\nafinder_app"
$env:DB_HOST="localhost"; $env:DB_PORT="3306"; $env:DB_NAME="nafinder"; $env:DB_USER="root"; $env:DB_PASS=""
python app.py
```

Biarkan jendela ini menyala. Buka browser ke http://127.0.0.1:5000, daftar akun, lalu login.

## Jalankan pipeline

Buka menu **Kelola Dataset**. Unggah CSV mentah `ulasan_wfc_final.csv`, jalankan **Preprocessing dan pelabelan**, lalu setelah selesai jalankan **Klasifikasi MNB manual**. Panel status menunjukkan tahap aktif, progres terukur, waktu berjalan, dan waktu total. Jangan tutup terminal/browser sebelum proses selesai. Hasil muncul di Dashboard dan Hasil Klasifikasi.

Dataset mentah harus memiliki kolom: `placeId`, `nama_kafe`, `teks_ulasan`, `rating_ulasan`, `tanggal_ulasan`, `kafe_id`, `valid`. Jangan unggah CSV hasil preprocessing untuk opsi proses dari awal. Setelah preprocessing berhasil, CSV mentah staging dihapus karena data dan cache label sudah tersimpan.

## Menghentikan aplikasi

Tekan `Ctrl+C` di PowerShell tempat Flask berjalan. Laragon dapat dihentikan terpisah.

## File kode yang menjelaskan metode

- `processing/text_processing.py` — label lexicon/aspek dan enam tahap preprocessing (Sastrawi hanya untuk stemming).
- `processing/naive_bayes.py` — TF-IDF manual, random oversampling manual, MNB manual.
- `processing/evaluation.py` — confusion matrix dan metrik manual.
- `processing/pipeline.py` — split, urutan pemanggilan proses, dan evaluasi.
- `import_data.py` — simpan hasil ke MySQL.
- `notebooks/` — tempat notebook pengujian dan analisis perbandingan metode.
