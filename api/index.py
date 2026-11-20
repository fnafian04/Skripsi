import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("VERCEL", "1")

from app import create_app

app = create_app()
_orig_wsgi = app.wsgi_app


def _vercel_wsgi_wrapper(environ, start_response):
    path = environ.get("PATH_INFO", "/")
    if path in ("/api/index", "/api/index.py"):
        orig = environ.get("HTTP_X_VERCEL_ORIGINAL_PATH") or "/"
        environ["PATH_INFO"] = orig if orig not in ("/api/index", "/api/index.py") else "/"
    elif path.startswith("/api/index/"):
        environ["PATH_INFO"] = path[len("/api/index"):] or "/"
    return _orig_wsgi(environ, start_response)


app.wsgi_app = _vercel_wsgi_wrapper
handler = app
