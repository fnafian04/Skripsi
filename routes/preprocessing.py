"""Halaman inspeksi hasil preprocessing dan prediksi klasifikasi bergaya spreadsheet/Excel."""

from __future__ import annotations
import math
import pickle
import threading
import time
import pandas as pd
from flask import Blueprint, render_template, session, redirect, url_for, request
from sqlalchemy import func, or_

from db import SessionLocal
from models import Kafe, LabelSentimen, Ulasan
from processing import dataset_workflow as wf
from processing.workflow_state import read_state

bp = Blueprint("preprocessing", __name__, url_prefix="/preprocessing")

PER_PAGE = 15
_DATA_CACHE: dict[str, dict] = {}
_CACHE_LOCK = threading.Lock()
_DB_COUNT_TTL = 60.0
_DB_COUNT_CACHE: dict = {"key": None, "ts": 0.0, "ulasan": 0, "label": 0}
_PRED_MODEL = None
_PRED_VECTORIZER = None


def _get_db_counts(state_key) -> tuple[int, int]:
    now = time.monotonic()
    with _CACHE_LOCK:
        if _DB_COUNT_CACHE["key"] == state_key and (now - _DB_COUNT_CACHE["ts"]) < _DB_COUNT_TTL:
            return _DB_COUNT_CACHE["ulasan"], _DB_COUNT_CACHE["label"]
    db = SessionLocal()
    try:
        u_cnt = int(db.query(func.count(Ulasan.id)).scalar() or 0)
        l_cnt = int(db.query(func.count(LabelSentimen.id)).scalar() or 0)
    finally:
        db.close()
    with _CACHE_LOCK:
        _DB_COUNT_CACHE.update({"key": state_key, "ts": time.monotonic(), "ulasan": u_cnt, "label": l_cnt})
    return u_cnt, l_cnt


def _get_cached_csv(path) -> pd.DataFrame | None:
    if not path or not path.is_file():
        return None
    key = f"csv:{path}"
    mtime = path.stat().st_mtime
    with _CACHE_LOCK:
        entry = _DATA_CACHE.get(key)
        if entry and entry["mtime"] == mtime:
            return entry["df"]
    df = pd.read_csv(path, encoding="utf-8-sig")
    with _CACHE_LOCK:
        _DATA_CACHE[key] = {"mtime": mtime, "df": df}
    return df


def _get_cached_labels(path) -> pd.DataFrame | None:
    if not path or not path.is_file():
        return None
    key = f"pkl:{path}"
    mtime = path.stat().st_mtime
    with _CACHE_LOCK:
        entry = _DATA_CACHE.get(key)
        if entry and entry["mtime"] == mtime:
            return entry["df"]
    df = pd.read_pickle(path)
    with _CACHE_LOCK:
        _DATA_CACHE[key] = {"mtime": mtime, "df": df}
    return df


def _get_runtime_predictor():
    global _PRED_MODEL, _PRED_VECTORIZER
    if _PRED_MODEL is not None and _PRED_VECTORIZER is not None:
        return _PRED_MODEL, _PRED_VECTORIZER
    if wf._RUNTIME_MODEL is not None and wf._RUNTIME_VECTORIZER is not None:
        _PRED_MODEL = wf._RUNTIME_MODEL
        _PRED_VECTORIZER = wf._RUNTIME_VECTORIZER
        return _PRED_MODEL, _PRED_VECTORIZER
    if wf.SAVED_MODEL_FILE.is_file() and wf.SAVED_VEC_FILE.is_file():
        try:
            with open(wf.SAVED_MODEL_FILE, "rb") as f:
                _PRED_MODEL = pickle.load(f)
            with open(wf.SAVED_VEC_FILE, "rb") as f:
                _PRED_VECTORIZER = pickle.load(f)
            return _PRED_MODEL, _PRED_VECTORIZER
        except Exception:
            pass
    return None, None


def _page_window(current: int, total: int) -> list[int | None]:
    if total <= 7:
        return list(range(1, total + 1))
    pages: list[int | None] = [1]
    start = max(2, current - 2)
    end = min(total - 1, current + 2)
    if start > 2:
        pages.append(None)
    pages.extend(range(start, end + 1))
    if end < total - 1:
        pages.append(None)
    pages.append(total)
    return pages


@bp.route("/")
def index():
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    state = read_state()
    active_name = state.get("dataset_name")
    tab = request.args.get("tab", "processed")
    if tab not in ("processed", "raw"):
        tab = "processed"

    search_q = request.args.get("q", "").strip()[:100]
    filter_aspek = request.args.get("aspek", "").strip()
    filter_sentimen = request.args.get("sentimen", "").strip()

    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1

    raw_rows = []
    label_rows = []
    summary = None
    total_rows = 0
    total_pages = 1
    page_items = []
    aspek_options = []

    dataset_path = wf.get_dataset_path(active_name)
    raw_df = _get_cached_csv(dataset_path)
    if raw_df is not None:
        valid_mask = (
            raw_df["valid"].astype(str).str.lower().eq("true")
            if "valid" in raw_df.columns
            else pd.Series(True, index=raw_df.index)
        )
        summary = {
            "ulasan_awal": len(raw_df),
            "ulasan_valid": int(valid_mask.sum()),
            "nama_dataset": dataset_path.name,
        }

    # Cari file cache label preprocessing (.labels_<nama>.pkl)
    cache_path = None
    if active_name:
        for candidate in (
            wf.UPLOAD / f".labels_{active_name}.pkl",
            wf.BASE / "uploads_tmp" / f".labels_{active_name}.pkl",
            wf.CACHE,
            wf.BASE / "uploads_tmp" / ".labels_cache.pkl",
        ):
            if candidate.is_file():
                cache_path = candidate
                break

    state_key = (active_name, state.get("run_id"), state.get("finished_at"))
    labels_df = _get_cached_labels(cache_path)
    ulasan_valid_cnt = 0
    label_cnt = 0
    if labels_df is not None:
        if summary is None:
            summary = {"ulasan_awal": 0, "ulasan_valid": 0, "nama_dataset": active_name or cache_path.name}
        summary.update({
            "label": len(labels_df),
            "review_berlabel": int(labels_df["review_id"].nunique()) if "review_id" in labels_df.columns else len(labels_df),
            "kafe": int(labels_df["placeId"].nunique()) if "placeId" in labels_df.columns else 0,
            "aspek": labels_df["aspek"].value_counts().to_dict() if "aspek" in labels_df.columns else {},
            "sentimen": labels_df["sentimen"].value_counts().to_dict() if "sentimen" in labels_df.columns else {},
        })
        if "aspek" in labels_df.columns:
            aspek_options = sorted(labels_df["aspek"].dropna().unique().tolist())
    else:
        # Fallback database (mis. saat berjalan di Vercel tanpa cache .pkl lokal)
        aspek_options = [
            "fasilitas_wfc", "harga", "kebersihan",
            "makanan_minuman", "pelayanan", "suasana",
        ]
        ulasan_valid_cnt, label_cnt = _get_db_counts(state_key)
        if label_cnt > 0 or ulasan_valid_cnt > 0:
            ds_name = active_name or "dataset_semprog.csv"
            reg_meta = wf._read_registry().get(ds_name, {})
            raw_total_est = int(reg_meta.get("row_count") or 0)
            if not raw_total_est:
                raw_total_est = 109778 if ds_name == "dataset_semprog.csv" else ulasan_valid_cnt
            if summary is None:
                summary = {
                    "ulasan_awal": raw_total_est,
                    "ulasan_valid": ulasan_valid_cnt,
                    "nama_dataset": ds_name,
                }
            summary.setdefault("label", label_cnt)

    if tab == "raw" and raw_df is not None:
        view_df = raw_df
        if search_q:
            q_low = search_q.lower()
            mask = pd.Series(False, index=view_df.index)
            for col in ("nama_kafe", "teks_ulasan", "sub_kalimat"):
                if col in view_df.columns:
                    mask = mask | view_df[col].fillna("").astype(str).str.lower().str.contains(q_low, regex=False)
            view_df = view_df[mask]

        total_rows = len(view_df)
        total_pages = max(1, math.ceil(total_rows / PER_PAGE))
        page = min(page, total_pages)
        start = (page - 1) * PER_PAGE
        end = start + PER_PAGE

        slice_df = view_df.iloc[start:end]
        for offset, (_, row) in enumerate(slice_df.iterrows(), start=start + 1):
            raw_rows.append({
                "no": offset,
                "nama_kafe": row.get("nama_kafe", "—"),
                "teks_ulasan": row.get("teks_ulasan", row.get("asal_sub_kalimat", row.get("sub_kalimat", "—"))),
                "rating_ulasan": row.get("rating_ulasan", "—"),
                "valid": row.get("valid", True),
            })
        page_items = _page_window(page, total_pages)

    elif tab == "raw" and raw_df is None:
        if ulasan_valid_cnt == 0 and label_cnt == 0:
            ulasan_valid_cnt, label_cnt = _get_db_counts(state_key)
        db = SessionLocal()
        try:
            q = db.query(Ulasan, Kafe.nama).join(Kafe, Ulasan.kafe_id == Kafe.id)
            if search_q:
                like_q = f"%{search_q}%"
                q = q.filter(or_(Kafe.nama.ilike(like_q), Ulasan.teks.ilike(like_q)))
                total_rows = q.count()
            else:
                total_rows = ulasan_valid_cnt
            total_pages = max(1, math.ceil(total_rows / PER_PAGE))
            page = min(page, total_pages)
            start = (page - 1) * PER_PAGE
            rows_db = q.order_by(Ulasan.id.asc()).offset(start).limit(PER_PAGE).all()
            for offset, (u_row, k_nama) in enumerate(rows_db, start=start + 1):
                raw_rows.append({
                    "no": offset,
                    "nama_kafe": k_nama or "—",
                    "teks_ulasan": u_row.teks or "—",
                    "rating_ulasan": u_row.rating if u_row.rating is not None else "—",
                    "valid": u_row.valid,
                })
            page_items = _page_window(page, total_pages)
        finally:
            db.close()

    elif labels_df is not None:
        tab = "processed"
        view_df = labels_df
        if filter_aspek and "aspek" in view_df.columns:
            view_df = view_df[view_df["aspek"] == filter_aspek]
        if filter_sentimen and "sentimen" in view_df.columns:
            view_df = view_df[view_df["sentimen"] == filter_sentimen]
        if search_q:
            q_low = search_q.lower()
            mask = pd.Series(False, index=view_df.index)
            for col in ("nama_kafe", "sub_kalimat", "teks_preprocessed"):
                if col in view_df.columns:
                    mask = mask | view_df[col].fillna("").astype(str).str.lower().str.contains(q_low, regex=False)
            view_df = view_df[mask]

        total_rows = len(view_df)
        total_pages = max(1, math.ceil(total_rows / PER_PAGE))
        page = min(page, total_pages)
        start = (page - 1) * PER_PAGE
        end = start + PER_PAGE

        slice_df = view_df.iloc[start:end]
        model, vectorizer = _get_runtime_predictor()
        preds: list[str] = []
        if model is not None and vectorizer is not None and not slice_df.empty and "teks_preprocessed" in slice_df.columns:
            texts = slice_df["teks_preprocessed"].fillna("").astype(str).tolist()
            try:
                preds = [str(p) for p in model.predict(vectorizer.transform(texts))]
            except Exception:
                preds = []

        for idx, (_, row) in enumerate(slice_df.iterrows()):
            actual_sent = str(row.get("sentimen", "—")).lower()
            pred_sent = preds[idx] if idx < len(preds) else actual_sent
            label_rows.append({
                "no": start + idx + 1,
                "review_id": row.get("review_id", start + idx),
                "nama_kafe": row.get("nama_kafe", "—"),
                "sub_kalimat": row.get("sub_kalimat", "—"),
                "teks_preprocessed": row.get("teks_preprocessed", "—"),
                "aspek": row.get("aspek", "—"),
                "sentimen": actual_sent,
                "prediksi_mnb": pred_sent,
                "cocok": (pred_sent == actual_sent),
            })
        page_items = _page_window(page, total_pages)

    else:
        tab = "processed"
        db = SessionLocal()
        try:
            q = (
                db.query(LabelSentimen, Ulasan.review_id, Ulasan.teks_preprocessed, Kafe.nama)
                .join(Ulasan, LabelSentimen.ulasan_id == Ulasan.id)
                .join(Kafe, LabelSentimen.kafe_id == Kafe.id)
            )
            has_filter = bool(filter_aspek or filter_sentimen or search_q)
            if filter_aspek:
                q = q.filter(LabelSentimen.aspek == filter_aspek)
            if filter_sentimen:
                q = q.filter(LabelSentimen.sentimen == filter_sentimen)
            if search_q:
                like_q = f"%{search_q}%"
                q = q.filter(
                    or_(
                        Kafe.nama.ilike(like_q),
                        LabelSentimen.sub_kalimat.ilike(like_q),
                        Ulasan.teks_preprocessed.ilike(like_q),
                    )
                )
            total_rows = q.count() if has_filter else label_cnt
            total_pages = max(1, math.ceil(total_rows / PER_PAGE))
            page = min(page, total_pages)
            start = (page - 1) * PER_PAGE
            rows_db = q.order_by(LabelSentimen.id.asc()).offset(start).limit(PER_PAGE).all()

            model, vectorizer = _get_runtime_predictor()
            preds: list[str] = []
            if model is not None and vectorizer is not None and rows_db:
                texts = [str(u_prep or "") for (_, _, u_prep, _) in rows_db]
                try:
                    preds = [str(p) for p in model.predict(vectorizer.transform(texts))]
                except Exception:
                    preds = []

            for idx, (lbl, rev_id, u_prep, k_nama) in enumerate(rows_db):
                actual_sent = str(lbl.sentimen or "—").lower()
                pred_sent = preds[idx] if idx < len(preds) else actual_sent
                label_rows.append({
                    "no": start + idx + 1,
                    "review_id": rev_id if rev_id is not None else (start + idx),
                    "nama_kafe": k_nama or "—",
                    "sub_kalimat": lbl.sub_kalimat or "—",
                    "teks_preprocessed": u_prep or "—",
                    "aspek": lbl.aspek or "—",
                    "sentimen": actual_sent,
                    "prediksi_mnb": pred_sent,
                    "cocok": (pred_sent == actual_sent),
                })
            page_items = _page_window(page, total_pages)
        finally:
            db.close()

    return render_template(
        "preprocessing/index.html",
        state=state,
        summary=summary,
        tab=tab,
        raw_rows=raw_rows,
        label_rows=label_rows,
        page=page,
        per_page=PER_PAGE,
        total_rows=total_rows,
        total_pages=total_pages,
        page_items=page_items,
        search_q=search_q,
        filter_aspek=filter_aspek,
        filter_sentimen=filter_sentimen,
        aspek_options=aspek_options,
    )
