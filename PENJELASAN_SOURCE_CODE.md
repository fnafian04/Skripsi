# Penjelasan Source Code Nafinder

Dokumen ini menjelaskan hubungan antara tahapan penelitian, file program, dan
perhitungan metode. Aplikasi utama menjalankan proses secara mandiri; notebook
dan script validasi hanya digunakan sebagai pembanding eksperimen.

## 1. Alur utama program

```text
CSV ulasan mentah
    ↓
processing/preprocessing.py
    ↓
processing/text_processing.py
    ↓
Data berlabel aspek dan sentimen
    ↓
processing/classification.py
    ↓
processing/split.py
    ↓
processing/naive_bayes.py
    ↓
processing/evaluation.py
    ↓
Metrik dan confusion matrix
```

Orkestrasi keseluruhan dilakukan oleh `processing/pipeline.py`. File ini
memanggil preprocessing terlebih dahulu, kemudian klasifikasi. Dengan begitu,
file orkestrator tidak menyimpan rumus metode dan lebih mudah dibaca.

## 2. Preprocessing dan pelabelan

### `processing/text_processing.py`

File ini berisi aturan dasar pengolahan teks:

1. memotong template boilerplate Google Maps;
2. memecah ulasan menjadi sub-kalimat berdasarkan tanda baca dan konjungsi kontras;
3. mendeteksi aspek menggunakan kamus kata kunci;
4. menghitung skor lexicon;
5. melakukan cleaning, case folding, tokenizing, normalisasi,
   stopword removal, dan stemming.

Skor lexicon dihitung dengan logika:

```text
skor = jumlah kata positif - jumlah kata negatif
```

Aturan juga mengenali beberapa frasa evaluatif dan negasi yang umum pada
ulasan kafe. Jika skor lebih besar dari nol, labelnya positif. Jika lebih kecil
dari nol, labelnya negatif. Jika skor sama dengan nol, rating digunakan sebagai cadangan:
rating 5 menjadi positif, rating 1 menjadi negatif, dan rating lainnya netral.

### `processing/preprocessing.py`

File ini menerapkan aturan pada seluruh dataset:

1. memvalidasi kolom CSV;
2. memberi identitas `review_id`;
3. memfilter baris dengan `valid == true`;
4. memanggil `proses_ulasan()` untuk setiap ulasan;
5. menyimpan aspek, sentimen, dan teks hasil preprocessing;
6. membuat ringkasan jumlah ulasan, label, kafe, dan distribusi kelas.

## 3. Pembagian data

### `processing/split.py`

Data dibagi berdasarkan `review_id`, bukan dengan mengacak setiap sub-kalimat
secara bebas. Konfigurasi aplikasi adalah:

```text
test_size = 0.2
random_state = 42
```

Setelah split, teks preprocessing yang identik dengan teks training juga
dikeluarkan dari testing untuk mengurangi kebocoran fitur.

## 4. Perhitungan fitur dan model

### `processing/naive_bayes.py`

#### TF-IDF manual

Program tidak memanggil `TfidfVectorizer`. Perhitungannya dibuat sendiri:

```text
tf(t,d) = 1 + ln(count(t,d))       jika count > 0
idf(t)  = ln((1 + N)/(1 + df(t))) + 1
tfidf   = tf × idf
```

Setiap vektor dokumen kemudian dinormalisasi dengan norma L2:

```text
v_norm = v / sqrt(sum(v_i²))
```

#### Random Oversampling manual

`RandomOverSamplerManual` hanya bekerja pada data training. Sampel kelas
minoritas dipilih ulang secara acak sampai jumlahnya sama dengan kelas
mayoritas. Data testing tidak diubah.

#### Multinomial Naive Bayes manual

`MultinomialNaiveBayesManual` menghitung:

```text
log P(c) = ln(Nc / N)

P(w|c) = (count(w,c) + alpha) /
         (total_count(c) + alpha × |V|)
```

Pipeline aplikasi menggunakan `alpha = 2.0`. Untuk setiap dokumen, skor kelas
dihitung dengan:

```text
score(c) = log P(c) + Σ tfidf(w,d) × log P(w|c)
```

Kelas dengan skor terbesar dipilih sebagai prediksi:

```text
ŷ = argmax score(c)
```

## 5. Evaluasi

### `processing/evaluation.py`

Program membangun confusion matrix dengan baris sebagai label aktual dan kolom
sebagai label prediksi. Dari matriks tersebut dihitung:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 × precision × recall / (precision + recall)
accuracy  = jumlah diagonal / jumlah seluruh data
```

Program juga menghitung macro average dan weighted average.

## 6. File yang memanggil proses

### `processing/classification.py`

Mengatur urutan split, TF-IDF, oversampling, training MNB, prediksi, dan
evaluasi. File ini memanggil implementasi rumus dari `naive_bayes.py`,
`split.py`, dan `evaluation.py`.

### `processing/pipeline.py`

Menjalankan:

```python
labels, summary = preprocess_and_label(raw)
metrics, model, vec, split = fit_evaluate_manual(labels)
```

### `processing/dataset_workflow.py`

Menghubungkan pipeline dengan aplikasi Flask, cache label, database, dan status
progress. Route tidak menghitung rumus secara langsung.

## 7. Manual atau library?

Model utama aplikasi adalah manual:

- TF-IDF manual;
- Random Oversampling manual;
- Multinomial Naive Bayes manual;
- evaluasi manual.

`numpy` dipakai untuk operasi array dan matriks, `pandas` untuk pengolahan
DataFrame, dan Sastrawi hanya digunakan untuk stemming. Scikit-learn dan
imbalanced-learn tidak diperlukan ketika aplikasi utama berjalan. Library
tersebut hanya muncul pada notebook atau script audit untuk membandingkan hasil
implementasi manual.

## 8. Cara menjelaskan saat presentasi

> “Source code dipisahkan berdasarkan tanggung jawab. `text_processing.py`
> berisi aturan preprocessing, `preprocessing.py` menerapkannya ke dataset,
> `classification.py` mengatur alur klasifikasi, sedangkan `naive_bayes.py`,
> `split.py`, dan `evaluation.py` berisi perhitungan metode. `pipeline.py`
> hanya memanggil tahap-tahap tersebut secara berurutan. Jadi model berjalan
> mandiri di dalam program dan tidak mengambil angka hasil dari notebook.”
