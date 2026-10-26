"""Riwayat eksperimen dan validasi manual labeling."""
import json
from pathlib import Path

from flask import Blueprint, redirect, render_template, session, url_for
from sqlalchemy import func

from db import SessionLocal
from models import ValidasiLabel, ValidasiV4, ValidasiV5

bp = Blueprint("riwayat_labeling", __name__, url_prefix="/riwayat-labeling")
ROOT = Path(__file__).resolve().parents[1]
UPLOADS = ROOT / "uploads_tmp"


def _read_json(filename):
    path = UPLOADS / filename
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _manual_counts(db, model):
    total = db.query(func.count(model.id)).scalar() or 0
    return {
        "total": total,
        "benar": db.query(func.count(model.id)).filter(model.status == "benar").scalar() or 0,
        "salah": db.query(func.count(model.id)).filter(model.status == "salah").scalar() or 0,
        "ragu": db.query(func.count(model.id)).filter(model.status == "ragu").scalar() or 0,
    }


@bp.get("/")
def index():
    if "user_id" not in session:
        return redirect(url_for("auth.login"))
    db = SessionLocal()
    try:
        baseline = _read_json("comparison_baseline_v5.json").get("baseline", {})
        v5 = _read_json("comparison_baseline_v5.json").get("v5", {})
        v4 = _read_json("labels_experiment_v4_summary.json")
        versions = [
            {
                "version": "v1",
                "name": "Baseline",
                "description": "Rule-based labeling awal yang digunakan sebagai pembanding utama",
                "manual": _manual_counts(db, ValidasiLabel),
                "metrics": baseline,
                "evidence": "Sudah divalidasi manual",
            },
            {
                "version": "v2",
                "name": "Eksperimen v2",
                "description": "Aturan berdasarkan pola kesalahan dari validasi manual sebelumnya",
                "manual": _manual_counts(db, ValidasiV4),
                "metrics": v4.get("metrics_alpha_2_min_df_1", {}),
                "evidence": "Sudah divalidasi manual",
            },
            {
                "version": "v3",
                "name": "Eksperimen v3",
                "description": "Perbaikan konteks informatif, WFC, pelayanan, dan keluhan tersirat",
                "manual": _manual_counts(db, ValidasiV5),
                "metrics": v5,
                "evidence": "Sudah divalidasi manual",
            },
        ]
        return render_template("riwayat_labeling/index.html", versions=versions)
    finally:
        db.close()
