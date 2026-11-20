"""Blueprint rekomendasi WFC — filter fasilitas + ranking dari skor_aspek."""

import threading
import time
from flask import Blueprint, render_template, session, redirect, url_for, request
from db import SessionLocal
from models import Kafe, SkorAspek, Ulasan
from processing.workflow_state import read_state
from sqlalchemy import func

bp = Blueprint('rekomendasi', __name__, url_prefix='/rekomendasi')

_REK_TTL = 60.0
_REK_LOCK = threading.Lock()
_REK_CACHE: dict = {'key': None, 'ts': 0.0, 'ranked': None, 'total_ulasan': 0}


def _login_required():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return None


def _load_ranked_catalog(state_key):
    now = time.monotonic()
    with _REK_LOCK:
        if _REK_CACHE['ranked'] is not None and _REK_CACHE['key'] == state_key and (now - _REK_CACHE['ts']) < _REK_TTL:
            return _REK_CACHE['ranked'], _REK_CACHE['total_ulasan']

    db = SessionLocal()
    try:
        # Hitung ulasan per kafe dalam 1 kali agregasi (menghindari scan berulang tabel ulasan)
        ulasan_counts = {
            int(kafe_id): int(cnt)
            for kafe_id, cnt in db.query(Ulasan.kafe_id, func.count(Ulasan.id)).group_by(Ulasan.kafe_id).all()
        }
        total_ulasan_tersimpan = sum(ulasan_counts.values())
        if not ulasan_counts:
            ranked_list = []
        else:
            active_kafes = db.query(Kafe).filter(Kafe.id.in_(list(ulasan_counts.keys()))).all()
            skor_records = db.query(SkorAspek).filter(SkorAspek.kafe_id.in_(list(ulasan_counts.keys()))).all()

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

            ranked_list = []
            for k in active_kafes:
                s_data = kafe_skor_map.get(
                    k.id, {'aspek_dict': {}, 'kekuatan': [], 'kelemahan': [], 'breakdown': []}
                )
                wfc_pos = s_data['aspek_dict'].get('fasilitas_wfc')
                wfc_fallback_used = False
                if wfc_pos is None:
                    wfc_fallback_used = True
                    all_pcts = list(s_data['aspek_dict'].values())
                    wfc_pos = sum(all_pcts) / len(all_pcts) if all_pcts else 0.0

                rating_gmaps = float(k.rating_gmaps) if k.rating_gmaps is not None else 0.0
                rating_norm = (rating_gmaps / 5.0) * 100.0
                poin_wfc = round(wfc_pos * 0.7, 1)
                poin_rating = round(rating_norm * 0.3, 1)
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

            ranked_list.sort(key=lambda x: (x['skor_akhir'], x['rating_gmaps'], x['jumlah_ulasan']), reverse=True)

        with _REK_LOCK:
            _REK_CACHE.update({
                'key': state_key,
                'ts': time.monotonic(),
                'ranked': ranked_list,
                'total_ulasan': total_ulasan_tersimpan,
            })
        return ranked_list, total_ulasan_tersimpan
    finally:
        db.close()


@bp.route('/')
def index():
    guard = _login_required()
    if guard:
        return guard

    filter_24jam = request.args.get('buka_24jam') == '1'
    filter_wifi = request.args.get('wifi') == '1'
    filter_stopkontak = request.args.get('stopkontak') == '1'
    filter_meja = request.args.get('meja_kerja') == '1'
    search_input = request.args.get('q', '').strip()[:100]
    search = search_input if len(search_input) >= 3 else ''

    state = read_state()
    state_key = (state.get('dataset_name'), state.get('run_id'), state.get('finished_at'))
    all_ranked, total_ulasan_tersimpan = _load_ranked_catalog(state_key)

    q_low = search.lower()
    matched = []
    for item in all_ranked:
        if filter_24jam and not item['buka_24jam']:
            continue
        if filter_wifi and not item['wifi']:
            continue
        if filter_stopkontak and not item['stopkontak']:
            continue
        if filter_meja and not item['meja_kerja']:
            continue
        if q_low and (
            q_low not in item['nama'].lower()
            and q_low not in item['alamat'].lower()
            and q_low not in (item['place_id'] or '').lower()
        ):
            continue
        matched.append(item)

    context = {
        'kafe_list': matched[:50],
        'total_cocok': len(matched),
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
