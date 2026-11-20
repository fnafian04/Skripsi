import os
import sys
from pathlib import Path

# Tambahkan direktori root proyek ke sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Tandai environment Vercel
os.environ.setdefault("VERCEL", "1")

from app import create_app

# WSGI Application entrypoint untuk Vercel Serverless Function
app = create_app()
handler = app
