"""Halaman ringkasan hasil preprocessing dataset aktif."""

import pandas as pd
from flask import Blueprint, render_template, session, redirect, url_for

from processing import dataset_workflow as wf
from processing.workflow_state import read_state

bp = Blueprint("preprocessing", __name__, url_prefix="/preprocessing")


@bp.route("/")
def index():
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    state = read_state()
    raw_sample = []
    label_sample = []
    summary = None
    dataset_path = wf.get_dataset_path(state.get("dataset_name"))
    if dataset_path and dataset_path.is_file():
        raw = pd.read_csv(dataset_path, encoding="utf-8-sig")
        valid = raw["valid"].astype(str).str.lower().eq("true") if "valid" in raw else pd.Series(True, index=raw.index)
        raw_sample = raw.loc[:, [c for c in ("nama_kafe", "teks_ulasan", "rating_ulasan") if c in raw]].head(10).to_dict("records")
        summary = {
            "ulasan_awal": len(raw),
            "ulasan_valid": int(valid.sum()),
            "nama_dataset": dataset_path.name,
        }

    cache = wf.UPLOAD / f".labels_{state.get('dataset_name')}.pkl" if state.get("dataset_name") else None
    if cache and cache.is_file():
        labels = pd.read_pickle(cache)
        label_sample = labels.loc[:, [c for c in ("nama_kafe", "sub_kalimat", "aspek", "sentimen", "teks_preprocessed") if c in labels]].head(10).to_dict("records")
        if summary is None:
            summary = {"ulasan_awal": 0, "ulasan_valid": 0, "nama_dataset": cache.name}
        summary.update({
            "label": len(labels),
            "review_berlabel": int(labels["review_id"].nunique()),
            "kafe": int(labels["placeId"].nunique()),
            "aspek": labels["aspek"].value_counts().to_dict(),
            "sentimen": labels["sentimen"].value_counts().to_dict(),
        })

    return render_template(
        "preprocessing/index.html",
        state=state,
        summary=summary,
        raw_sample=raw_sample,
        label_sample=label_sample,
    )
