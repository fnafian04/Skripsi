-- ==================================================================
-- NAFINDER — Skema Database MySQL
-- Rekomendasi Kafe WFC Kota Malang | Multinomial Naive Bayes
-- Fadhli Ilham Nafi'an Yuswono (23.18.001) — ITN Malang
--
-- Jalankan sekali untuk membuat seluruh struktur:
--   mysql -u root -p < schema.sql
--
-- Prinsip penamaan (dipakai konsisten di seluruh file):
--   - tabel jamak, huruf kecil: users, kafe, ulasan
--   - kolom huruf kecil + garis bawah: teks_preprocessed
--   - kunci asing selalu berakhiran _id
--   - ENUM membatasi nilai di level database, bukan hanya aplikasi
-- ==================================================================

CREATE DATABASE IF NOT EXISTS nafinder
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;
USE nafinder;

SET FOREIGN_KEY_CHECKS = 0;
DROP TABLE IF EXISTS riwayat_upload, evaluasi_model, skor_aspek,
                     label_sentimen, ulasan, kafe, users;
SET FOREIGN_KEY_CHECKS = 1;

-- ==================================================================
-- 1. USERS — autentikasi (login/register)
-- ==================================================================
CREATE TABLE users (
  id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  username      VARCHAR(50)  NOT NULL UNIQUE,
  email         VARCHAR(120) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,          -- hash bcrypt, bukan teks polos
  dibuat_pada   TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- ==================================================================
-- 2. KAFE — master kafe; kolom boolean = fasilitas untuk hard filter
-- ==================================================================
CREATE TABLE kafe (
  id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  place_id      VARCHAR(128) NOT NULL UNIQUE,   -- placeId dari Google Maps
  nama          VARCHAR(160) NOT NULL,
  alamat        VARCHAR(255) NULL,
  rating_gmaps  DECIMAL(2,1) NULL,              -- 1.0 s/d 5.0
  url_maps      VARCHAR(512) NULL,
  jumlah_ulasan INT UNSIGNED NOT NULL DEFAULT 0,
  -- 4 filter fasilitas WFC (sesuai Bab III); NULL = data belum diketahui
  buka_24jam    TINYINT(1)   NULL,
  wifi          TINYINT(1)   NULL,
  stopkontak    TINYINT(1)   NULL,
  meja_kerja    TINYINT(1)   NULL,
  INDEX idx_fasilitas (buka_24jam, wifi, stopkontak, meja_kerja)
) ENGINE=InnoDB;

-- ==================================================================
-- 3. ULASAN — data mentah hasil upload CSV, satu baris satu ulasan
-- ==================================================================
CREATE TABLE ulasan (
  id               BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  kafe_id          INT UNSIGNED NOT NULL,
  review_id        BIGINT UNSIGNED NOT NULL,    -- penanda asli di CSV sumber
  teks             TEXT          NOT NULL,
  rating           TINYINT UNSIGNED NULL,       -- 1..5
  tanggal          DATE          NULL,
  teks_preprocessed TEXT         NULL,          -- hasil 6 tahap preprocessing
  valid            TINYINT(1)    NOT NULL DEFAULT 1,
  CONSTRAINT fk_ulasan_kafe FOREIGN KEY (kafe_id)
    REFERENCES kafe (id) ON DELETE CASCADE,
  UNIQUE KEY uq_ulasan (kafe_id, review_id),
  INDEX idx_ulasan_kafe (kafe_id)
) ENGINE=InnoDB;

-- ==================================================================
-- 4. LABEL_SENTIMEN — inti analisis: per sub-kalimat per aspek
--    Satu baris = satu label, sama seperti dataset_preprocessed.csv
--    (240.636 label pada run 15 Okt 2026)
-- ==================================================================
CREATE TABLE label_sentimen (
  id          BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  ulasan_id   BIGINT UNSIGNED NOT NULL,
  kafe_id     INT UNSIGNED    NOT NULL,         -- di-deret agar agregasi per kafe cepat
  sub_kalimat VARCHAR(2048)   NOT NULL,
  aspek       ENUM('makanan_minuman', 'pelayanan', 'harga',
                   'suasana', 'fasilitas_wfc', 'kebersihan') NOT NULL,
  sentimen    ENUM('positif', 'netral', 'negatif') NOT NULL,
  CONSTRAINT fk_label_ulasan FOREIGN KEY (ulasan_id)
    REFERENCES ulasan (id) ON DELETE CASCADE,
  CONSTRAINT fk_label_kafe   FOREIGN KEY (kafe_id)
    REFERENCES kafe (id)   ON DELETE CASCADE,
  INDEX idx_label_kafe_aspek (kafe_id, aspek, sentimen)
) ENGINE=InnoDB;

-- ==================================================================
-- 5. SKOR_ASPEK — hasil agregasi label_sentimen per kafe per aspek.
--    Disimpan (materialisasi) supaya dashboard & rekomendasi tidak
--    menghitung ulang 240 ribu label tiap kali halaman dibuka.
--    Dihitung ulang hanya saat dataset baru diunggah.
-- ==================================================================
CREATE TABLE skor_aspek (
  kafe_id        INT UNSIGNED NOT NULL,
  aspek          ENUM('makanan_minuman', 'pelayanan', 'harga',
                      'suasana', 'fasilitas_wfc', 'kebersihan') NOT NULL,
  jumlah_positif INT UNSIGNED NOT NULL DEFAULT 0,
  jumlah_netral  INT UNSIGNED NOT NULL DEFAULT 0,
  jumlah_negatif INT UNSIGNED NOT NULL DEFAULT 0,
  persen_positif DECIMAL(5,2) NOT NULL DEFAULT 0,
  persen_negatif DECIMAL(5,2) NOT NULL DEFAULT 0,
  label_kekuatan TINYINT(1)   NOT NULL DEFAULT 0,  -- persen_positif >= 60 (Bab III)
  label_kelemahan TINYINT(1)  NOT NULL DEFAULT 0,  -- persen_negatif >= 30 (Bab III)
  dihitung_pada  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
                              ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (kafe_id, aspek),
  CONSTRAINT fk_skor_kafe FOREIGN KEY (kafe_id)
    REFERENCES kafe (id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ==================================================================
-- 6. EVALUASI_MODEL — ringkasan metrik untuk kartu dashboard
--    & halaman hasil klasifikasi (hasil run Colab disimpan di sini)
-- ==================================================================
CREATE TABLE evaluasi_model (
  id             INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  nama_model     VARCHAR(60)  NOT NULL DEFAULT 'Multinomial Naive Bayes',
  accuracy       DECIMAL(5,4) NOT NULL,          -- 0.6983 = 69,83%
  precision_macro DECIMAL(5,4) NOT NULL,
  recall_macro   DECIMAL(5,4) NOT NULL,
  f1_macro       DECIMAL(5,4) NOT NULL,
  f1_weighted    DECIMAL(5,4) NOT NULL,
  jumlah_testing INT UNSIGNED NOT NULL,           -- 37.548
  confusion_matrix JSON NULL,                     -- 3x3 matriks sebagai JSON
  file_model     VARCHAR(120) NULL,               -- nama berkas .pkl
  dilatih_pada   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- ==================================================================
-- 7. RIWAYAT_UPLOAD — jejak tiap unggahan dataset (audit & debugging)
-- ==================================================================
CREATE TABLE riwayat_upload (
  id           INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  nama_file    VARCHAR(160) NOT NULL,
  jumlah_baris INT UNSIGNED NOT NULL DEFAULT 0,
  jumlah_kafe  INT UNSIGNED NOT NULL DEFAULT 0,
  status       ENUM('berhasil', 'gagal', 'sebagian') NOT NULL,
  pesan        VARCHAR(255) NULL,
  user_id      INT UNSIGNED NULL,
  waktu        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_upload_user FOREIGN KEY (user_id)
    REFERENCES users (id) ON DELETE SET NULL
) ENGINE=InnoDB;

-- ==================================================================
-- TABEL JANGKAUAN (opsional, untuk pengembangan berikutnya):
-- Fasilitas tidak selalu 4 itu — bila kelak bertambah (musala,
-- area merokok, parkir), pindahkan ke tabel ini supaya tidak perlu
-- ALTER TABLE pada tabel kafe yang sudah penuh data.
-- Kolom boolean di atas TETAP dipakai untuk query filter cepat.
-- ==================================================================
-- CREATE TABLE fasilitas (
--   kafe_id  INT UNSIGNED NOT NULL,
--   nama     VARCHAR(40)   NOT NULL,   -- 'wifi', 'buka_24jam', ...
--   tersedia TINYINT(1)    NOT NULL DEFAULT 0,
--   PRIMARY KEY (kafe_id, nama),
--   FOREIGN KEY (kafe_id) REFERENCES kafe (id) ON DELETE CASCADE
-- );

-- 4b. VALIDASI_LABEL — checklist manual untuk sampel label, tanpa mengubah label asli
CREATE TABLE validasi_label (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    label_id BIGINT UNSIGNED NOT NULL,
    urutan INT UNSIGNED NOT NULL,
    status VARCHAR(16) NULL,
    aspek_koreksi VARCHAR(32) NULL,
    sentimen_koreksi VARCHAR(16) NULL,
    catatan VARCHAR(500) NULL,
    user_id INT UNSIGNED NULL,
    diperiksa_pada DATETIME NULL,
    CONSTRAINT uq_validasi_label UNIQUE (label_id),
    CONSTRAINT fk_validasi_label FOREIGN KEY (label_id)
        REFERENCES label_sentimen(id) ON DELETE CASCADE,
    CONSTRAINT fk_validasi_user FOREIGN KEY (user_id)
        REFERENCES users(id) ON DELETE SET NULL,
    INDEX idx_validasi_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE validasi_label_aspek (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    validasi_id BIGINT UNSIGNED NOT NULL,
    aspek ENUM('makanan_minuman','pelayanan','harga','suasana','fasilitas_wfc','kebersihan') NOT NULL,
    CONSTRAINT uq_validasi_aspek UNIQUE (validasi_id, aspek),
    CONSTRAINT fk_validasi_aspek FOREIGN KEY (validasi_id)
        REFERENCES validasi_label(id) ON DELETE CASCADE,
    INDEX idx_validasi_aspek (validasi_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE validasi_v4 (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    urutan INT UNSIGNED NOT NULL,
    sub_kalimat VARCHAR(2048) NOT NULL,
    teks_ulasan TEXT NOT NULL,
    rating TINYINT UNSIGNED NULL,
    aspek ENUM('makanan_minuman','pelayanan','harga','suasana','fasilitas_wfc','kebersihan') NOT NULL,
    sentimen ENUM('positif','netral','negatif') NOT NULL,
    status VARCHAR(16) NULL,
    sentimen_koreksi VARCHAR(16) NULL,
    catatan VARCHAR(500) NULL,
    diperiksa_pada DATETIME NULL,
    INDEX idx_validasi_v4_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE validasi_v4_aspek (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    validasi_id BIGINT UNSIGNED NOT NULL,
    aspek ENUM('makanan_minuman','pelayanan','harga','suasana','fasilitas_wfc','kebersihan') NOT NULL,
    CONSTRAINT uq_validasi_v4_aspek UNIQUE (validasi_id, aspek),
    CONSTRAINT fk_validasi_v4_aspek FOREIGN KEY (validasi_id)
        REFERENCES validasi_v4(id) ON DELETE CASCADE,
    INDEX idx_validasi_v4_aspek (validasi_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE validasi_v5 (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    urutan INT UNSIGNED NOT NULL,
    sub_kalimat VARCHAR(2048) NOT NULL,
    teks_ulasan TEXT NOT NULL,
    rating TINYINT UNSIGNED NULL,
    aspek ENUM('makanan_minuman','pelayanan','harga','suasana','fasilitas_wfc','kebersihan') NOT NULL,
    sentimen ENUM('positif','netral','negatif') NOT NULL,
    status VARCHAR(16) NULL,
    sentimen_koreksi VARCHAR(16) NULL,
    catatan VARCHAR(500) NULL,
    diperiksa_pada DATETIME NULL,
    INDEX idx_validasi_v5_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE validasi_v5_aspek (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    validasi_id BIGINT UNSIGNED NOT NULL,
    aspek ENUM('makanan_minuman','pelayanan','harga','suasana','fasilitas_wfc','kebersihan') NOT NULL,
    CONSTRAINT uq_validasi_v5_aspek UNIQUE (validasi_id, aspek),
    CONSTRAINT fk_validasi_v5_aspek FOREIGN KEY (validasi_id)
        REFERENCES validasi_v5(id) ON DELETE CASCADE,
    INDEX idx_validasi_v5_aspek (validasi_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 4c. LABEL_SENTIMEN_OPTIMIZED — hasil eksperimen aturan preprocessing baru
CREATE TABLE label_sentimen_optimized (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    ulasan_id BIGINT UNSIGNED NOT NULL,
    kafe_id INT UNSIGNED NOT NULL,
    sub_kalimat VARCHAR(2048) NOT NULL,
    aspek ENUM('makanan_minuman','pelayanan','harga','suasana','fasilitas_wfc','kebersihan') NOT NULL,
    sentimen ENUM('positif','netral','negatif') NOT NULL,
    CONSTRAINT fk_label_opt_ulasan FOREIGN KEY (ulasan_id)
        REFERENCES ulasan(id) ON DELETE CASCADE,
    CONSTRAINT fk_label_opt_kafe FOREIGN KEY (kafe_id)
        REFERENCES kafe(id) ON DELETE CASCADE,
    INDEX idx_label_opt_kafe_aspek (kafe_id, aspek, sentimen)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
