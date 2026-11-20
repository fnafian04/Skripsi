"""Blueprint rekomendasi WFC — filter fasilitas + ranking dari skor_aspek."""

from flask import Blueprint, render_template, session, redirect, url_for, request
from db import SessionLocal
from models import Kafe, SkorAspek, Ulasan
from sqlalchemy import func

bp = Blueprint('rekomendasi', __name__, url_prefix='/rekomendasi')


def _login_required():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return None


@bp.route('/')
def index():
    guard = _login_required()
    if guard:
        return guard

    # Nilai filter dari query string: ?buka_24jam=1&wifi=1&...
    filter_24jam = request.args.get('buka_24jam') == '1'
    filter_wifi = request.args.get('wifi') == '1'
    filter_stopkontak = request.args.get('stopkontak') == '1'
    filter_meja = request.args.get('meja_kerja') == '1'
    search_input = request.args.get('q', '').strip()[:100]
    search = search_input if len(search_input) >= 3 else ''

    db = SessionLocal()
    try:
        # 1. Base query filter fasilitas
        # Master kafe dipertahankan saat dataset dihapus. Rekomendasi hanya
        # boleh membaca kafe yang masih memiliki ulasan pada dataset aktif.
        query = db.query(Kafe).join(Ulasan, Ulasan.kafe_id == Kafe.id).distinct()
        if filter_24jam:
            query = query.filter(Kafe.buka_24jam.is_(True))
        if filter_wifi:
            query = query.filter(Kafe.wifi.is_(True))
        if filter_stopkontak:
            query = query.filter(Kafe.stopkontak.is_(True))
        if filter_meja:
            query = query.filter(Kafe.meja_kerja.is_(True))
        if search:
            escaped_search = search.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
            pattern = f'%{escaped_search}%'
            query = query.filter(
                Kafe.nama.ilike(pattern, escape='\\')
                | Kafe.alamat.ilike(pattern, escape='\\')
                | Kafe.place_id.ilike(pattern, escape='\\')
            )

        total_ulasan_tersimpan = db.query(func.count(Ulasan.id)).scalar() or 0
        filtered_kafes = query.all()
        if not filtered_kafes:
            context = {
                'kafe_list': [],
                'total_cocok': 0,
                'total_ulasan_tersimpan': total_ulasan_tersimpan,
                'f': {'buka_24jam': filter_24jam, 'wifi': filter_wifi, 'stopkontak': filter_stopkontak, 'meja_kerja': filter_meja},
                'search': search,
            }
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return render_template('rekomendasi/_results.html', **context)
            return render_template('rekomendasi/index.html', **context)

        kafe_ids = [k.id for k in filtered_kafes]
        ulasan_counts = dict(
            db.query(Ulasan.kafe_id, func.count(Ulasan.id))
            .filter(Ulasan.kafe_id.in_(kafe_ids))
            .group_by(Ulasan.kafe_id)
            .all()
        )

        # 2. Ambil skor_aspek untuk semua kafe yang terfilter
        skor_records = db.query(SkorAspek).filter(SkorAspek.kafe_id.in_(kafe_ids)).all()
        
        # Kelompokkan skor aspek per kafe
        kafe_skor_map = {}
        for s in skor_records:
            if s.kafe_id not in kafe_skor_map:
                kafe_skor_map[s.kafe_id] = {
                    'aspek_dict': {},
                    'kekuatan': [],
                    'kelemahan': [],
                    'breakdown': [],
                }
            pos_pct = float(s.persen_positif or 0)
            pos_cnt = int(s.jumlah_positif or 0)
            net_cnt = int(s.jumlah_netral or 0)
            neg_cnt = int(s.jumlah_negatif or 0)
            tot_cnt = pos_cnt + net_cnt + neg_cnt
            kafe_skor_map[s.kafe_id]['aspek_dict'][s.aspek] = pos_pct
            kafe_skor_map[s.kafe_id]['breakdown'].append({
                'aspek': s.aspek,
                'label': s.aspek.replace('_', ' ').title(),
                'positif': pos_cnt,
                'negatif': neg_cnt,
                'total': tot_cnt,
                'persen_positif': round(pos_pct, 1),
            })
            if s.label_kekuatan:
                kafe_skor_map[s.kafe_id]['kekuatan'].append(s.aspek.replace('_', ' ').title())
            if s.label_kelemahan:
                kafe_skor_map[s.kafe_id]['kelemahan'].append(s.aspek.replace('_', ' ').title())

        # 3. Hitung skor rekomendasi per kafe
        # Bobot rekomendasi: 70% sentimen positif fasilitas WFC + 30% rating
        # yang tersimpan bersama ulasan dataset aktif.
        ranked_list = []
        for k in filtered_kafes:
            s_data = kafe_skor_map.get(
                k.id, {'aspek_dict': {}, 'kekuatan': [], 'kelemahan': [], 'breakdown': []}
            )

            # Persentase positif aspek fasilitas_wfc
            wfc_pos = s_data['aspek_dict'].get('fasilitas_wfc')
            wfc_fallback_used = False
            if wfc_pos is None:
                # Fallback: rerata sentimen positif semua aspek jika belum ada ulasan WFC khusus
                wfc_fallback_used = True
                all_pcts = list(s_data['aspek_dict'].values())
                wfc_pos = sum(all_pcts) / len(all_pcts) if all_pcts else 0.0

            rating_gmaps = float(k.rating_gmaps) if k.rating_gmaps is not None else 0.0
            rating_norm = (rating_gmaps / 5.0) * 100.0

            poin_wfc = round(wfc_pos * 0.7, 1)
            poin_rating = round(rating_norm * 0.3, 1)
            # Formula Skor Rekomendasi Nafinder
            skor_akhir = round((wfc_pos * 0.7) + (rating_norm * 0.3), 1)
            breakdown_sorted = sorted(
                s_data['breakdown'],
                key=lambda item: (item['positif'], item['persen_positif']),
                reverse=True,
            )

            ranked_list.append({
                'id': k.id,
                'place_id': k.place_id,
                'url_maps': f'https://www.google.com/maps/place/?q=place_id:{k.place_id}',
                'nama': k.nama,
                'alamat': k.alamat or '—',
                'rating_gmaps': rating_gmaps,
                'jumlah_ulasan': int(ulasan_counts.get(k.id, 0)),
                'persen_wfc': round(wfc_pos, 1),
                'wfc_fallback_used': wfc_fallback_used,
                'poin_wfc': poin_wfc,
                'poin_rating': poin_rating,
                'skor_akhir': skor_akhir,
                'buka_24jam': bool(k.buka_24jam),
                'wifi': bool(k.wifi),
                'stopkontak': bool(k.stopkontak),
                'meja_kerja': bool(k.meja_kerja),
                'kekuatan': s_data['kekuatan'],
                'kelemahan': s_data['kelemahan'],
                'aspek_breakdown': breakdown_sorted,
            })

        # Urutkan berdasarkan skor_akhir tertinggi, lalu rating gmaps, lalu jumlah ulasan
        ranked_list.sort(key=lambda x: (x['skor_akhir'], x['rating_gmaps'], x['jumlah_ulasan']), reverse=True)

        context = {
            'kafe_list': ranked_list[:50],
            'total_cocok': len(ranked_list),
            'total_ulasan_tersimpan': total_ulasan_tersimpan,
            'f': {
                'buka_24jam': filter_24jam,
                'wifi': filter_wifi,
                'stopkontak': filter_stopkontak,
                'meja_kerja': filter_meja,
            },
            'search': search,
        }
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return render_template('rekomendasi/_results.html', **context)
        return render_template('rekomendasi/index.html', **context)
    finally:
        db.close()
