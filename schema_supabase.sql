-- ==================================================================
-- NAFINDER - Skema Database Supabase PostgreSQL
-- Rekomendasi Kafe WFC Kota Malang | Multinomial Naive Bayes
-- Fadhli Ilham Nafi'an Yuswono (23.18.001) - ITN Malang
--
-- Jalankan skrip ini di SQL Editor pada dashboard Supabase
-- ==================================================================

DROP TABLE IF EXISTS validasi_v5_aspek CASCADE;
DROP TABLE IF EXISTS validasi_v5 CASCADE;
DROP TABLE IF EXISTS validasi_v4_aspek CASCADE;
DROP TABLE IF EXISTS validasi_v4 CASCADE;
DROP TABLE IF EXISTS validasi_label_aspek CASCADE;
DROP TABLE IF EXISTS validasi_label CASCADE;
DROP TABLE IF EXISTS riwayat_upload CASCADE;
DROP TABLE IF EXISTS evaluasi_model CASCADE;
DROP TABLE IF EXISTS skor_aspek CASCADE;
DROP TABLE IF EXISTS label_sentimen_optimized CASCADE;
DROP TABLE IF EXISTS label_sentimen CASCADE;
DROP TABLE IF EXISTS ulasan CASCADE;
DROP TABLE IF EXISTS kafe CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- 1. USERS
CREATE TABLE users (
  id            SERIAL PRIMARY KEY,
  username      VARCHAR(50) NOT NULL UNIQUE,
  email         VARCHAR(120) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  dibuat_pada   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 2. KAFE
CREATE TABLE kafe (
  id            SERIAL PRIMARY KEY,
  place_id      VARCHAR(128) NOT NULL UNIQUE,
  nama          VARCHAR(160) NOT NULL,
  alamat        VARCHAR(255) NULL,
  rating_gmaps  NUMERIC(2,1) NULL,
  url_maps      VARCHAR(512) NULL,
  jumlah_ulasan INTEGER NOT NULL DEFAULT 0,
  buka_24jam    BOOLEAN NULL,
  wifi          BOOLEAN NULL,
  stopkontak    BOOLEAN NULL,
  meja_kerja    BOOLEAN NULL
);
CREATE INDEX idx_kafe_fasilitas ON kafe(buka_24jam, wifi, stopkontak, meja_kerja);

-- 3. ULASAN
CREATE TABLE ulasan (
  id                BIGSERIAL PRIMARY KEY,
  kafe_id           INTEGER NOT NULL REFERENCES kafe(id) ON DELETE CASCADE,
  review_id         BIGINT NOT NULL,
  teks              TEXT NOT NULL,
  rating            SMALLINT NULL,
  tanggal           DATE NULL,
  teks_preprocessed TEXT NULL,
  valid             BOOLEAN NOT NULL DEFAULT TRUE,
  CONSTRAINT uq_ulasan UNIQUE (kafe_id, review_id)
);
CREATE INDEX idx_ulasan_kafe ON ulasan(kafe_id);

-- 4. LABEL_SENTIMEN
CREATE TABLE label_sentimen (
  id          BIGSERIAL PRIMARY KEY,
  ulasan_id   BIGINT NOT NULL REFERENCES ulasan(id) ON DELETE CASCADE,
  kafe_id     INTEGER NOT NULL REFERENCES kafe(id) ON DELETE CASCADE,
  sub_kalimat VARCHAR(2048) NOT NULL,
  aspek       VARCHAR(50) NOT NULL CHECK (aspek IN ('makanan_minuman', 'pelayanan', 'harga', 'suasana', 'fasilitas_wfc', 'kebersihan')),
  sentimen    VARCHAR(20) NOT NULL CHECK (sentimen IN ('positif', 'netral', 'negatif'))
);
CREATE INDEX idx_label_kafe_aspek ON label_sentimen(kafe_id, aspek, sentimen);

-- 4b. LABEL_SENTIMEN_OPTIMIZED
CREATE TABLE label_sentimen_optimized (
  id          BIGSERIAL PRIMARY KEY,
  ulasan_id   BIGINT NOT NULL REFERENCES ulasan(id) ON DELETE CASCADE,
  kafe_id     INTEGER NOT NULL REFERENCES kafe(id) ON DELETE CASCADE,
  sub_kalimat VARCHAR(2048) NOT NULL,
  aspek       VARCHAR(50) NOT NULL CHECK (aspek IN ('makanan_minuman', 'pelayanan', 'harga', 'suasana', 'fasilitas_wfc', 'kebersihan')),
  sentimen    VARCHAR(20) NOT NULL CHECK (sentimen IN ('positif', 'netral', 'negatif'))
);
CREATE INDEX idx_label_opt_kafe_aspek ON label_sentimen_optimized(kafe_id, aspek, sentimen);

-- 5. SKOR_ASPEK
CREATE TABLE skor_aspek (
  kafe_id        INTEGER NOT NULL REFERENCES kafe(id) ON DELETE CASCADE,
  aspek          VARCHAR(50) NOT NULL CHECK (aspek IN ('makanan_minuman', 'pelayanan', 'harga', 'suasana', 'fasilitas_wfc', 'kebersihan')),
  jumlah_positif INTEGER NOT NULL DEFAULT 0,
  jumlah_netral  INTEGER NOT NULL DEFAULT 0,
  jumlah_negatif INTEGER NOT NULL DEFAULT 0,
  persen_positif NUMERIC(5,2) NOT NULL DEFAULT 0,
  persen_negatif NUMERIC(5,2) NOT NULL DEFAULT 0,
  label_kekuatan BOOLEAN NOT NULL DEFAULT FALSE,
  label_kelemahan BOOLEAN NOT NULL DEFAULT FALSE,
  dihitung_pada  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (kafe_id, aspek)
);

-- 6. EVALUASI_MODEL
CREATE TABLE evaluasi_model (
  id              SERIAL PRIMARY KEY,
  nama_model      VARCHAR(60) NOT NULL DEFAULT 'Multinomial Naive Bayes',
  accuracy        NUMERIC(5,4) NOT NULL,
  precision_macro NUMERIC(5,4) NOT NULL,
  recall_macro    NUMERIC(5,4) NOT NULL,
  f1_macro        NUMERIC(5,4) NOT NULL,
  f1_weighted     NUMERIC(5,4) NOT NULL,
  jumlah_testing  INTEGER NOT NULL,
  confusion_matrix JSONB NULL,
  file_model      VARCHAR(120) NULL,
  dilatih_pada    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 7. RIWAYAT_UPLOAD
CREATE TABLE riwayat_upload (
  id           SERIAL PRIMARY KEY,
  nama_file    VARCHAR(160) NOT NULL,
  jumlah_baris INTEGER NOT NULL DEFAULT 0,
  jumlah_kafe  INTEGER NOT NULL DEFAULT 0,
  status       VARCHAR(20) NOT NULL CHECK (status IN ('berhasil', 'gagal', 'sebagian')),
  pesan        VARCHAR(255) NULL,
  user_id      INTEGER NULL REFERENCES users(id) ON DELETE SET NULL,
  waktu        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 8. VALIDASI_V4 & V5
CREATE TABLE validasi_v4 (
  id               SERIAL PRIMARY KEY,
  urutan           INTEGER NOT NULL,
  sub_kalimat      VARCHAR(2048) NOT NULL,
  teks_ulasan      TEXT NULL,
  rating           SMALLINT NULL,
  aspek            VARCHAR(50) NOT NULL,
  sentimen         VARCHAR(20) NOT NULL,
  status           VARCHAR(50) NOT NULL,
  sentimen_koreksi VARCHAR(20) NULL,
  catatan          TEXT NULL,
  diperiksa_pada   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE validasi_v4_aspek (
  id          SERIAL PRIMARY KEY,
  validasi_id INTEGER NOT NULL REFERENCES validasi_v4(id) ON DELETE CASCADE,
  aspek       VARCHAR(50) NOT NULL
);

CREATE TABLE validasi_v5 (
  id               SERIAL PRIMARY KEY,
  urutan           INTEGER NOT NULL,
  sub_kalimat      VARCHAR(2048) NOT NULL,
  teks_ulasan      TEXT NULL,
  rating           SMALLINT NULL,
  aspek            VARCHAR(50) NOT NULL,
  sentimen         VARCHAR(20) NOT NULL,
  status           VARCHAR(50) NOT NULL,
  sentimen_koreksi VARCHAR(20) NULL,
  catatan          TEXT NULL,
  diperiksa_pada   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE validasi_v5_aspek (
  id          SERIAL PRIMARY KEY,
  validasi_id INTEGER NOT NULL REFERENCES validasi_v5(id) ON DELETE CASCADE,
  aspek       VARCHAR(50) NOT NULL
);


-- Nonaktifkan RLS agar semua modul aplikasi Flask dapat membaca dan menulis data
ALTER TABLE IF EXISTS users DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS kafe DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS ulasan DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS label_sentimen DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS label_sentimen_optimized DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS skor_aspek DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS evaluasi_model DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS riwayat_upload DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS validasi_v4 DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS validasi_v4_aspek DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS validasi_v5 DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS validasi_v5_aspek DISABLE ROW LEVEL SECURITY;

-- Berikan hak akses penuh ke role service
GRANT ALL ON ALL TABLES IN SCHEMA public TO postgres, anon, authenticated, service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO postgres, anon, authenticated, service_role;

