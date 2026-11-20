# Panduan mencoba Nafinder

## Menjalankan aplikasi

1. Nyalakan Laragon dengan **Start All**.
2. Dari PowerShell, jalankan aplikasi:

   ```powershell
   cd "D:\Skripsi\Road to Semhas\program"
   python app.py
   ```

3. Buka `http://127.0.0.1:5000` lalu masuk ke akun.
4. Setelah masuk, halaman **Bantuan & panduan** menjelaskan cara menggunakan sistem.

## Mencoba fitur pengguna

1. Buka **Rekomendasi kafe** untuk mencari kafe dan menyaringnya berdasarkan Wi-Fi, stopkontak, meja kerja, atau jam buka.
2. Buka **Uji model**, tulis ulasan seperti “Makanannya enak, tetapi tempatnya sempit dan Wi-Fi lambat”, lalu pilih **Analisis ulasan**.
3. Periksa aspek yang dikenali dan hasil sentimen positif atau negatifnya.
4. Di bawah fitur uji ulasan, lihat ringkasan hasil evaluasi model serta rincian prediksi yang benar dan keliru.

## Dataset penelitian

Aplikasi menggunakan satu dataset final berisi 2.000 label yang telah diperiksa anotator. Dataset ini dikunci agar hasil penelitian tidak berubah karena unggahan atau penggantian data. Hasil uji yang tersimpan adalah accuracy 90,33% pada 331 contoh testing.

Jangan menghapus berkas dataset, cache label, atau data evaluasi pada database jika aplikasi masih akan digunakan.
