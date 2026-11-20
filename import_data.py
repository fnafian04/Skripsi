"""Impor dataset ulasan mentah melalui pipeline Nafinder.
Preprocessing, pelabelan, TF-IDF manual, ROS manual, MNB manual, dan metrik
manual dijalankan oleh processing/pipeline.py sebelum transaksi database.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import pandas as pd
from sqlalchemy import text
from db import engine
from processing.pipeline import run_full_pipeline


def _parse_address(value):
    if not isinstance(value, str):
        return None
    return value.split('|', 1)[1].strip()[:255] if '|' in value else value.strip()[:255]


def _parse_date(value):
    """Ubah waktu relatif Google Maps menjadi tanggal perkiraan bila dikenali."""
    if not isinstance(value, str) or not value.strip():
        return None
    parts = value.split()
    if len(parts) < 2:
        return None
    try:
        amount = int(parts[0])
    except ValueError:
        return None
    days = {'hari': 1, 'minggu': 7, 'bulan': 30, 'tahun': 365}.get(parts[1].lower())
    if days is None:
        return None
    return (pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=amount * days)).date()


def run_import(filepath: str, user_id: int | None = None, progress=None) -> dict:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")
    result = run_full_pipeline(filepath, progress=progress)
    raw = pd.read_csv(filepath, encoding='utf-8-sig')
    raw['review_id'] = range(len(raw))
    raw['valid'] = raw['valid'].astype(str).str.lower().eq('true')
    raw = raw[raw.valid].copy()
    labels = result['labels']
    metrics = result['metrics']

    # Kafe master: satu entri per placeId, fasilitas ditandai bila disebut di ulasan.
    cafes = []
    for place_id, group in raw.groupby('placeId', sort=False):
        first = group.iloc[0]
        text_blob = ' '.join(group['teks_ulasan'].fillna('').astype(str)).lower()
        rates = pd.to_numeric(group['rating_ulasan'], errors='coerce').dropna()
        cafes.append({
            'place_id': str(place_id), 'nama': str(first['nama_kafe'])[:160],
            'alamat': _parse_address(first['kafe_id']),
            'rating_gmaps': round(float(rates.mean()), 1) if len(rates) else None,
            'jumlah_ulasan': int(group['review_id'].nunique()),
            'buka_24jam': int('24 jam' in text_blob or '24jam' in text_blob),
            'wifi': int('wifi' in text_blob or 'wi-fi' in text_blob or 'internet' in text_blob),
            'stopkontak': int(any(x in text_blob for x in ('colokan','stopkontak','stop kontak','charger'))),
            'meja_kerja': int(any(x in text_blob for x in ('meja','kursi','nugas','laptop','wfc','kerja'))),
        })
    cafe_ids = {}

    # Materialize source rows and associated model labels, replacing prior dataset atomically.
    with engine.begin() as conn:
        if engine.dialect.name == 'postgresql':
            cafe_sql = text("""
                INSERT INTO kafe (place_id,nama,alamat,rating_gmaps,jumlah_ulasan,buka_24jam,wifi,stopkontak,meja_kerja)
                VALUES (:place_id,:nama,:alamat,:rating_gmaps,:jumlah_ulasan,:buka_24jam,:wifi,:stopkontak,:meja_kerja)
                ON CONFLICT (place_id) DO UPDATE SET
                  nama=EXCLUDED.nama, alamat=EXCLUDED.alamat, rating_gmaps=EXCLUDED.rating_gmaps,
                  jumlah_ulasan=EXCLUDED.jumlah_ulasan, buka_24jam=EXCLUDED.buka_24jam,
                  wifi=EXCLUDED.wifi, stopkontak=EXCLUDED.stopkontak, meja_kerja=EXCLUDED.meja_kerja
            """)
        else:
            cafe_sql = text("""
                INSERT INTO kafe (place_id,nama,alamat,rating_gmaps,jumlah_ulasan,buka_24jam,wifi,stopkontak,meja_kerja)
                VALUES (:place_id,:nama,:alamat,:rating_gmaps,:jumlah_ulasan,:buka_24jam,:wifi,:stopkontak,:meja_kerja)
                ON DUPLICATE KEY UPDATE nama=VALUES(nama), alamat=VALUES(alamat), rating_gmaps=VALUES(rating_gmaps),
                  jumlah_ulasan=VALUES(jumlah_ulasan), buka_24jam=VALUES(buka_24jam), wifi=VALUES(wifi),
                  stopkontak=VALUES(stopkontak), meja_kerja=VALUES(meja_kerja)
            """)
        conn.execute(cafe_sql, cafes)
        ids = conn.execute(text('SELECT id, place_id FROM kafe')).fetchall()
        cafe_ids = {p: int(i) for i,p in ids}

        # Clear child tables in FK order; all changes are within one DB transaction.
        conn.execute(text('DELETE FROM label_sentimen'))
        conn.execute(text('DELETE FROM skor_aspek'))
        conn.execute(text('DELETE FROM ulasan'))

        review_rows = []
        for _, row in raw.drop_duplicates(['placeId','review_id']).iterrows():
            try: rating = int(round(float(row['rating_ulasan'])))
            except (TypeError, ValueError): rating = None
            if rating is not None: rating = max(1, min(5, rating))
            review_rows.append({
                'kafe_id': cafe_ids[str(row['placeId'])], 'review_id': int(row['review_id']),
                'teks': str(row['teks_ulasan'])[:60000], 'rating': rating,
                'tanggal': _parse_date(row.get('tanggal_ulasan')), 'teks_preprocessed': None,
            })
        insert_reviews = text("""
            INSERT INTO ulasan (kafe_id,review_id,teks,rating,tanggal,teks_preprocessed,valid)
            VALUES (:kafe_id,:review_id,:teks,:rating,:tanggal,:teks_preprocessed,1)
        """)
        for start in range(0, len(review_rows), 2000):
            conn.execute(insert_reviews, review_rows[start:start+2000])

        review_map_rows = conn.execute(text('SELECT id,kafe_id,review_id FROM ulasan')).fetchall()
        review_ids = {(int(k),int(r)): int(i) for i,k,r in review_map_rows}
        labels_by_review = {}
        for _, row in labels.iterrows():
            key = (cafe_ids[str(row.placeId)], int(row.review_id))
            uid = review_ids.get(key)
            if uid is None: continue
            labels_by_review.setdefault(uid, []).append({
                'ulasan_id': uid, 'kafe_id': key[0], 'sub_kalimat': str(row.sub_kalimat)[:512],
                'aspek': row.aspek, 'sentimen': row.sentimen,
            })
        label_rows = [item for chunk in labels_by_review.values() for item in chunk]
        insert_label = text("""
            INSERT INTO label_sentimen (ulasan_id,kafe_id,sub_kalimat,aspek,sentimen)
            VALUES (:ulasan_id,:kafe_id,:sub_kalimat,:aspek,:sentimen)
        """)
        for start in range(0, len(label_rows), 3000):
            conn.execute(insert_label, label_rows[start:start+3000])

        conn.execute(text("""
            INSERT INTO skor_aspek (kafe_id,aspek,jumlah_positif,jumlah_netral,jumlah_negatif,
              persen_positif,persen_negatif,label_kekuatan,label_kelemahan)
            SELECT kafe_id,aspek,
              SUM(CASE WHEN sentimen='positif' THEN 1 ELSE 0 END),
              SUM(CASE WHEN sentimen='netral' THEN 1 ELSE 0 END),
              SUM(CASE WHEN sentimen='negatif' THEN 1 ELSE 0 END),
              ROUND(100.0 * SUM(CASE WHEN sentimen='positif' THEN 1 ELSE 0 END) / COUNT(*), 2),
              ROUND(100.0 * SUM(CASE WHEN sentimen='negatif' THEN 1 ELSE 0 END) / COUNT(*), 2),
              CASE WHEN (100.0 * SUM(CASE WHEN sentimen='positif' THEN 1 ELSE 0 END) / COUNT(*)) >= 60 THEN TRUE ELSE FALSE END,
              CASE WHEN (100.0 * SUM(CASE WHEN sentimen='negatif' THEN 1 ELSE 0 END) / COUNT(*)) >= 30 THEN TRUE ELSE FALSE END
            FROM label_sentimen GROUP BY kafe_id,aspek
        """))

        cm_data = {
            'classes': metrics['classes'], 'matrix': metrics['matrix'],
            'per_class': metrics['per_class'],
        }
        conn.execute(text('DELETE FROM evaluasi_model'))
        conn.execute(text("""
            INSERT INTO evaluasi_model (nama_model,accuracy,precision_macro,recall_macro,f1_macro,
              f1_weighted,jumlah_testing,confusion_matrix,file_model)
            VALUES (:nama,:acc,:pm,:rm,:fm,:fw,:n,:cm,:file)
        """), {
            'nama': 'Multinomial Naive Bayes (manual)', 'acc': metrics['accuracy'],
            'pm': metrics['macro']['precision'], 'rm': metrics['macro']['recall'],
            'fm': metrics['macro']['f1'], 'fw': metrics['weighted']['f1'],
            'n': metrics['jumlah_testing'], 'cm': json.dumps(cm_data), 'file': None,
        })
        conn.execute(text("""
            INSERT INTO riwayat_upload (nama_file,jumlah_baris,jumlah_kafe,status,pesan,user_id)
            VALUES (:filename,:rows,:cafes,'berhasil',:message,:user_id)
        """), {
            'filename': Path(filepath).name, 'rows': len(raw), 'cafes': len(cafes),
            'message': f"Pipeline manual selesai: {len(cafes)} kafe, {len(review_rows)} ulasan, {len(label_rows)} label; akurasi {metrics['accuracy']:.4f}, macro-F1 {metrics['macro']['f1']:.4f}.",
            'user_id': user_id,
        })

    return {'status':'success','kafe_count':len(cafes),'ulasan_count':len(review_rows),
            'label_count':len(label_rows),'metrics':metrics,'preprocessing':result['preprocessing']}


if __name__ == '__main__':
    run_import(os.path.join('uploads_tmp','datasets','dataset_semprog.csv'))
