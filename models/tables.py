"""
Model SQLAlchemy Nafinder — kompatibel dengan MySQL (Laragon) dan PostgreSQL (Supabase).

Tipe data integer menggunakan tipe standar SQLAlchemy (Integer, BigInteger, SmallInteger)
agar portabel lintas dialect database tanpa error 'unsigned' di PostgreSQL.
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Text, Enum, Boolean,
    Numeric, Date, DateTime, JSON, ForeignKey, UniqueConstraint, Index,
    Integer, BigInteger, SmallInteger
)
from db import Base

# Tipe data integer generik yang kompatibel untuk MySQL dan PostgreSQL
UInt = Integer               # INT / SERIAL
UBigInt = BigInteger         # BIGINT / BIGSERIAL
UTinyInt = SmallInteger      # SMALLINT / TINYINT (rating 1..5)


# Enum dengan native_enum=False agar disimpan sebagai VARCHAR + CHECK constraint
# sehingga portabel dan tidak bentrok antar dialect database
ASPEK_ENUM = Enum(
    'makanan_minuman', 'pelayanan', 'harga',
    'suasana', 'fasilitas_wfc', 'kebersihan',
    name='aspek_enum',
    native_enum=False
)
SENTIMEN_ENUM = Enum('positif', 'netral', 'negatif', name='sentimen_enum', native_enum=False)
STATUS_UPLOAD_ENUM = Enum('berhasil', 'gagal', 'sebagian', name='status_upload_enum', native_enum=False)


class User(Base):
    __tablename__ = 'users'

    id = Column(UInt, primary_key=True, autoincrement=True)
    username = Column(String(50), nullable=False, unique=True)
    email = Column(String(120), nullable=False, unique=True)
    password_hash = Column(String(255), nullable=False)
    dibuat_pada = Column(DateTime, nullable=False, default=datetime.utcnow)


class Kafe(Base):
    __tablename__ = 'kafe'

    id = Column(UInt, primary_key=True, autoincrement=True)
    place_id = Column(String(128), nullable=False, unique=True)
    nama = Column(String(160), nullable=False)
    alamat = Column(String(255), nullable=True)
    rating_gmaps = Column(Numeric(2, 1), nullable=True)
    url_maps = Column(String(512), nullable=True)
    jumlah_ulasan = Column(UInt, nullable=False, default=0)

    # 4 fasilitas WFC — None = belum diketahui, False/0 = tidak tersedia, True/1 = tersedia
    buka_24jam = Column(Boolean, nullable=True)
    wifi = Column(Boolean, nullable=True)
    stopkontak = Column(Boolean, nullable=True)
    meja_kerja = Column(Boolean, nullable=True)

    __table_args__ = (
        Index('idx_fasilitas', 'buka_24jam', 'wifi', 'stopkontak', 'meja_kerja'),
    )


class Ulasan(Base):
    __tablename__ = 'ulasan'
    __table_args__ = (
        UniqueConstraint('kafe_id', 'review_id', name='uq_ulasan'),
        Index('idx_ulasan_kafe', 'kafe_id'),
    )

    id = Column(UBigInt, primary_key=True, autoincrement=True)
    kafe_id = Column(UInt, ForeignKey('kafe.id', ondelete='CASCADE'), nullable=False)
    review_id = Column(UBigInt, nullable=False)
    teks = Column(Text, nullable=False)
    rating = Column(UTinyInt, nullable=True)   # 1..5
    tanggal = Column(Date, nullable=True)
    teks_preprocessed = Column(Text, nullable=True)
    valid = Column(Boolean, nullable=False, default=True)


class LabelSentimen(Base):
    __tablename__ = 'label_sentimen'
    __table_args__ = (
        Index('idx_label_kafe_aspek', 'kafe_id', 'aspek', 'sentimen'),
    )

    id = Column(UBigInt, primary_key=True, autoincrement=True)
    ulasan_id = Column(UBigInt, ForeignKey('ulasan.id', ondelete='CASCADE'), nullable=False)
    kafe_id = Column(UInt, ForeignKey('kafe.id', ondelete='CASCADE'), nullable=False)
    sub_kalimat = Column(String(2048), nullable=False)
    aspek = Column(ASPEK_ENUM, nullable=False)
    sentimen = Column(SENTIMEN_ENUM, nullable=False)


class LabelSentimenOptimized(Base):
    __tablename__ = 'label_sentimen_optimized'
    __table_args__ = (
        Index('idx_label_opt_kafe_aspek', 'kafe_id', 'aspek', 'sentimen'),
    )

    id = Column(UBigInt, primary_key=True, autoincrement=True)
    ulasan_id = Column(UBigInt, ForeignKey('ulasan.id', ondelete='CASCADE'), nullable=False)
    kafe_id = Column(UInt, ForeignKey('kafe.id', ondelete='CASCADE'), nullable=False)
    sub_kalimat = Column(String(2048), nullable=False)
    aspek = Column(ASPEK_ENUM, nullable=False)
    sentimen = Column(SENTIMEN_ENUM, nullable=False)


class SkorAspek(Base):
    __tablename__ = 'skor_aspek'

    kafe_id = Column(UInt, ForeignKey('kafe.id', ondelete='CASCADE'), primary_key=True)
    aspek = Column(ASPEK_ENUM, primary_key=True)

    jumlah_positif = Column(UInt, nullable=False, default=0)
    jumlah_netral = Column(UInt, nullable=False, default=0)
    jumlah_negatif = Column(UInt, nullable=False, default=0)
    persen_positif = Column(Numeric(5, 2), nullable=False, default=0)
    persen_negatif = Column(Numeric(5, 2), nullable=False, default=0)
    label_kekuatan = Column(Boolean, nullable=False, default=False)   # >= 60%
    label_kelemahan = Column(Boolean, nullable=False, default=False)  # >= 30%
    dihitung_pada = Column(DateTime, nullable=False, default=datetime.utcnow,
                           onupdate=datetime.utcnow)


class EvaluasiModel(Base):
    __tablename__ = 'evaluasi_model'

    id = Column(UInt, primary_key=True, autoincrement=True)
    nama_model = Column(String(60), nullable=False,
                        default='Multinomial Naive Bayes')
    accuracy = Column(Numeric(5, 4), nullable=False)
    precision_macro = Column(Numeric(5, 4), nullable=False)
    recall_macro = Column(Numeric(5, 4), nullable=False)
    f1_macro = Column(Numeric(5, 4), nullable=False)
    f1_weighted = Column(Numeric(5, 4), nullable=False)
    jumlah_testing = Column(UInt, nullable=False)
    confusion_matrix = Column(JSON, nullable=True)
    file_model = Column(String(120), nullable=True)
    dilatih_pada = Column(DateTime, nullable=False, default=datetime.utcnow)


class RiwayatUpload(Base):
    __tablename__ = 'riwayat_upload'

    id = Column(UInt, primary_key=True, autoincrement=True)
    nama_file = Column(String(160), nullable=False)
    jumlah_baris = Column(UInt, nullable=False, default=0)
    jumlah_kafe = Column(UInt, nullable=False, default=0)
    status = Column(STATUS_UPLOAD_ENUM, nullable=False)
    pesan = Column(String(255), nullable=True)
    user_id = Column(UInt, ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    waktu = Column(DateTime, nullable=False, default=datetime.utcnow)
