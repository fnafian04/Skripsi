"""Halaman dokumentasi rumus dan logika metode."""
from flask import Blueprint, redirect, render_template, session, url_for

bp = Blueprint("dokumentasi", __name__, url_prefix="/dokumentasi")


@bp.route("/")
def index():
    if "user_id" not in session:
        return redirect(url_for("auth.login"))
    return render_template("dokumentasi/index.html")
