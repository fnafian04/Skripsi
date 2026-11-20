# Nafinder — Aspect-Based Sentiment Analysis & WFC Cafe Recommendation

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![Database](https://img.shields.io/badge/database-MySQL-orange.svg)](https://www.mysql.com/)
[![License](https://img.shields.io/badge/license-Academic-green.svg)](#)

> **Sistem Rekomendasi Kafe Work-From-Cafe (WFC) di Kota Malang Berbasis Aspect-Based Sentiment Analysis (ABSA) Menggunakan Algoritma Multinomial Naive Bayes Manual.**  
> Proyek Penelitian Skripsi oleh **Nafii** ([@fnafian04](https://github.com/fnafian04)).

---

## 📌 Ringkasan Penelitian

Nafinder adalah sistem analitik dan rekomendasi kafe berbasis web yang mengekstrak ulasan publik Google Maps untuk mengukur sentimen pengunjung terhadap aspek-aspek vital *Work-From-Cafe* (WFC). Seluruh algoritma inti pemrosesan teks, pembobotan fitur, penyeimbangan data, hingga inferensi klasifikasi dibangun **secara manual (*from scratch*)** tanpa ketergantungan pada pustaka *machine learning* tingkat tinggi (*scikit-learn* atau *imbalanced-learn* pada aplikasi web).

### 🏷️ 6 Aspek Penelitian WFC:
1. **Makanan & Minuman** (`makanan_minuman`): Kualitas rasa, varian beans/biji kopi, pastry, susu, dan minuman.
2. **Pelayanan** (`pelayanan`): Keramahan staf, kecepatan barista, kasir, dan waktu tunggu pesanan.
3. **Harga** (`harga`): Tingkat keterjangkauan, kesesuaian harga (*worth it*), dan kisaran harga.
4. **Suasana** (`suasana`): Kenyamanan ruang, tingkat kebisingan, estetika, pencahayaan, dan *vibe* bekerja.
5. **Fasilitas WFC** (`fasilitas_wfc`): Ketersediaan stopkontak/colokan, stabilitas Wi-Fi, meja kerja, dan mushola.
6. **Kebersihan** (`kebersihan`): Higienitas area kafe, kebersihan meja, lantai, dan toilet.

---

## 🎯 Performa Model Hasil Anotasi

Model dievaluasi menggunakan pembagian data anti-kebocoran (**Group Shuffle Split 80:20 berbasis `review_id`**) dengan penyaringan sampel uji identik.

| Metrik Evaluasi | Nilai Capaian | Keterangan |
| :--- | :---: | :--- |
| **Akurasi Keseluruhan** | **90.33%** | Kemampuan klasifikasi pada 331 label data uji |
| **Macro F1-Score** | **0.9033** | Rata-rata F1 kelas negatif dan positif |
| **Weighted F1-Score** | **0.9033** | Rata-rata terbobot sesuai proporsi kelas |
| **Total Label Testing** | **331** | Sampel uji setelah penghapusan data bocor |
| **Total Kosakata TF-IDF** | **1.651** | Fitur kata unigram-bigram dari data latih |

### Confusion Matrix (Baseline MNB)
$$\begin{array}{c|cc}
\text{Aktual \ Prediksi} & \textbf{Negatif} & \textbf{Positif} \\
\hline
\textbf{Negatif} & 153 & 11 \\
\textbf{Positif} & 21 & 146 \\
\end{array}$$

---

## 🏗️ Alur & Arsitektur Sistem

```text
Google Maps Reviews (CSV Mentah)
            │
            ▼
┌──────────────────────────────────────────────┐
│  Tahap 0: Pemotongan Template Boilerplate    │
│  (Menghapus blok menu, jenis pesanan, dll.)  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Tahap 1: Pemecahan Sub-Kalimat              │
│  (Pemisah konjungsi pertentangan & koma)     │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Tahap 2: Deteksi Aspek & Labeling Lexicon   │
│  (6 Kamus Aspek + Aturan Negasi Leksikon)    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Tahap 3: Enam Tahap Text Preprocessing      │
│  1. Cleaning       4. Normalisasi            │
│  2. Case Folding   5. Stopword Removal       │
│  3. Tokenizing     6. Stemming (Sastrawi)    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Tahap 4: Pembagian Data Anti-Kebocoran      │
│  (GroupShuffleSplit 80:20 by review_id)      │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Tahap 5: Pembobotan Fitur TF-IDF Manual     │
│  (Unigram-bigram, Sublinear TF, Smooth IDF)  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Tahap 6: Data latih tanpa oversampling      │
│  (Konfigurasi model utama sesuai notebook)   │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Tahap 7: Multinomial Naive Bayes Manual     │
│  (Laplace Smoothing alpha=0.1 & MAP argmax)   │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Tahap 8: Evaluasi & Sistem Rekomendasi      │
│  (Agregasi Kekuatan Aspek Kafe di Web App)   │
└──────────────────────────────────────────────┘
```

---

## 💻 Implementasi Metode Manual (*From Scratch*)

Semua fungsi matematika inti ditulis tanpa *black-box* library:

1. **TF-IDF Manual (`processing/naive_bayes.py`)**:
   - **Sublinear TF**: $\text{tf}(t, d) = 1 + \ln(\text{count}(t, d))$ jika $\text{count} > 0$, selain itu $0$.
   - **Smooth IDF**: $\text{idf}(t) = \ln\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1.0$
   - **Normalisasi Euclidean (L2)**: $v_{\text{norm}} = \frac{v}{\sqrt{\sum v_i^2}}$
2. **Internal Sparse Matrix (`processing/sparse_matrix.py`)**:
   - Struktur data *Compressed Sparse Row* (CSR) internal untuk menghemat RAM saat memproses ratusan ribu dokumen tanpa SciPy.
3. **Random Oversampler Manual (`processing/naive_bayes.py`)**:
   - Eksperimen ROS dilakukan hanya pada data training; konfigurasi hasil utama memakai model tanpa ROS.
4. **Multinomial Naive Bayes Manual (`processing/naive_bayes.py`)**:
   - **Log Prior**: $\ln P(c) = \ln(N_c / N)$
   - **Log Likelihood (Laplace Smoothing $\alpha=0.1$)**:
     $$\ln P(w \mid c) = \ln\left(\frac{\sum \text{tf}(w, c) + \alpha}{\sum_{w'} \sum \text{tf}(w', c) + \alpha \cdot |V|}\right)$$
   - **Maximum A Posteriori (MAP)**:
     $$\hat{y} = \arg\max_{c} \left[ \ln P(c) + \sum_{w \in d} \text{tf-idf}(w, d) \cdot \ln P(w \mid c) \right]$$

---

## 📁 Struktur Direktori

```text
Program/
├── app.py                      # Entry point aplikasi Flask
├── db.py                       # Konfigurasi SQLAlchemy & koneksi database MySQL
├── schema.sql                  # Skema basis data lengkap
├── requirements.txt            # Dependensi paket Python
├── jalankan.bat                # Skrip eksekusi satu klik untuk Windows
├── CARA_MENJALANKAN.md         # Panduan instalasi dan deployment lokal
├── PENJELASAN_SOURCE_CODE.md   # Bedah teknis arsitektur kode
│
├── processing/                 # Engine komputasi NLP & Machine Learning Manual
│   ├── text_processing.py      # Regex, kamus lexicon, 6 tahap preprocessing
│   ├── preprocessing.py        # Pipeline ulasan mentah ke dataset berlabel
│   ├── classification.py       # Pipeline split, fit, predict, dan evaluasi
│   ├── naive_bayes.py          # Implementasi manual TF-IDF, ROS, dan MNB
│   ├── evaluation.py           # Kalkulasi Confusion Matrix, Precision, Recall, F1
│   ├── sparse_matrix.py        # Kelas CSR Matrix mandiri hemat memori
│   ├── split.py                # GroupShuffleSplit anti-data leakage
│   └── dataset_workflow.py     # Pengelola workflow multi-dataset & sinkronisasi DB
│
├── routes/                     # Blueprint kontroler aplikasi Flask
│   ├── auth.py                 # Autentikasi sesi pengguna
│   ├── dashboard.py            # Ringkasan analitik dan metrik kafe
│   ├── dataset.py              # Antarmuka unggah, preprocessing, dan klasifikasi
│   ├── hasil.py                # Visualisasi hasil evaluasi model & confusion matrix
│   ├── rekomendasi.py          # Pencarian kafe berdasarkan aspek WFC & fasilitas
│   └── dokumentasi.py          # Penjelasan rumus matematika interaktif
│
├── templates/                  # Templat tampilan Jinja2 (HTML5 semantik)
├── static/                     # Aset styling (Vanilla CSS modern) dan JavaScript
└── notebooks/                  # Notebook Jupyter eksperimen & komparasi model
    ├── preprocessing.ipynb     # Ground truth preprocessing 240k label
    ├── klasifikasi_mnb.ipynb   # Ground truth pemodelan baseline MNB
    ├── perbandingan_metode.ipynb # Komparasi 8 model pembelajaran mesin
    └── scraping.ipynb          # Skrip ekstraksi data Google Maps
```

---

## 🚀 Panduan Menjalankan Aplikasi

### 1. Prasyarat Sistem
- **Python**: Versi 3.10 atau lebih baru.
- **Database**: MySQL 8.0+ atau MariaDB (via Laragon / XAMPP).
- **Web Browser**: Chrome, Edge, Firefox, atau browser modern lainnya.

### 2. Kloning Repositori
```bash
git clone https://github.com/fnafian04/nafinder.git
cd nafinder
```

### 3. Persiapan Database
1. Buka MySQL client (HeidiSQL / phpMyAdmin / CLI).
2. Buat basis data baru:
   ```sql
   CREATE DATABASE nafinder CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   ```
3. Eksekusi berkas `schema.sql` ke dalam database `nafinder`.

### 4. Instalasi Dependensi
Buat virtual environment dan pasang pustaka yang diperlukan:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 5. Menjalankan Server
Jalankan aplikasi dengan perintah:
```bash
python app.py
```
*Atau di Windows cukup klik dua kali berkas `jalankan.bat`.*

Aplikasi akan aktif pada: **`http://127.0.0.1:5000`**  
*(Akun default: user `admin` / password tertera di `CARA_MENJALANKAN.md` atau register akun baru)*.

---

## 📊 Eksperimen Pembanding (*Ablation Study*)

Selain model baseline utama, eksperimen lanjutan disimpan sebagai bahan analisis internal:
- **Model final anotasi**: Klasifikasi biner dengan label hasil anotator, split berdasarkan `review_id` (Akurasi: 90.33%, Macro F1: 0.9033).
- **Eksperimen Pembanding (v2 & v3)**: Eksplorasi segmentasi klausa koordinatif dan pemisahan indikator konteks pelayanan untuk analisis variasi sebaran aspek.

---

## 👨‍💻 Peneliti & Pengembang

- **Nama**: Nafii
- **NIM**: 2318001
- **GitHub**: [@fnafian04](https://github.com/fnafian04)
- **Fokus**: Natural Language Processing (NLP), Aspect-Based Sentiment Analysis (ABSA), Information Retrieval.
