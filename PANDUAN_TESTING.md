# Langkah uji Nafinder — sebelum dataset penelitian

File uji: `uploads_tmp/ulasan_wfc_testing_5000.csv` (5.000 ulasan valid,
tersebar pada 324 kafe). File ini untuk memeriksa alur dan waktu proses saja;
hasil metriknya bukan hasil penelitian final dan tidak perlu sama dengan hasil
pengujian di Colab.

## Mulai

1. Nyalakan Laragon dengan **Start All**.
2. Jalankan `jalankan.bat` dari folder aplikasi, lalu buka `http://127.0.0.1:5000` dan login.
3. Buka **Kelola Dataset**. Pilih `ulasan_wfc_testing_5000.csv` dari folder `uploads_tmp` aplikasi, lalu **Unggah dataset baru**.
4. Klik **Mulai preprocessing**. Status, tahap berjalan, persen, dan waktu berjalan akan diperbarui pada panel bawah. Tunggu hingga status menunjukkan selesai.
5. Setelah tahap pertama selesai, tombol **Jalankan klasifikasi** akan aktif. Klik dan tunggu hasilnya; halaman Hasil Klasifikasi menampilkan metrik.
6. Catat waktu yang tertera dan kirim screenshot/error bila ada.

Untuk kembali ke keadaan bersih, gunakan **Hapus dataset dan hasil**. Tindakan ini permanen: menghapus staging upload, cache, label/ulasan/aspek, dan evaluasi aplikasi dari database; data master kafe dipertahankan. Jangan gunakan pada dataset penelitian yang ingin dipertahankan.

Sesudah alur pendek berhasil, unggah `ulasan_wfc_final.csv` untuk proses penuh. Berkas 5.000 baris adalah fixture uji dan tidak untuk angka skripsi.

## Perbandingan dengan Colab

Hasil aplikasi dan hasil Colab disimpan sebagai hasil eksperimen masing-masing.
Tidak ada angka Colab yang dipakai sebagai target aplikasi. Bandingkan
accuracy, precision, recall, F1, confusion matrix, jumlah baris, distribusi
kelas, hasil split, dan parameter. Jika berbeda, catat penyebabnya sebelum
menyimpulkan apakah implementasi perlu diperbaiki.
