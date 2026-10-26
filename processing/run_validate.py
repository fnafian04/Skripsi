"""Jalankan klasifikasi manual pada file label (.pkl) atau CSV preprocessed.

Berguna untuk mengecek angka metrik tanpa harus lewat antarmuka web.

Cara menjalankan:
    python processing/run_validate.py path\\ke\\csv_hasil_preprocessing.csv
    python processing/run_validate.py --pkl uploads_tmp/.labels_cache.pkl

CSV preprocessed harus diberikan secara eksplisit karena bukan file runtime
aplikasi dan tidak disimpan sebagai dataset aktif.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.classification import fit_evaluate_manual  # noqa: E402

def main() -> int:
    parser = argparse.ArgumentParser(description="Jalankan klasifikasi MNB manual.")
    parser.add_argument("csv", nargs="?", default=None,
                        help="Path CSV preprocessed (wajib jika tanpa --pkl).")
    parser.add_argument("--pkl", default=None,
                        help="Path file .pkl berisi DataFrame label.")
    args = parser.parse_args()

    if args.pkl:
        path = Path(args.pkl)
        print(f"sumber  : {path} (pickle)")
        df = pd.read_pickle(path)
    else:
        if not args.csv:
            parser.error("berikan path CSV preprocessed atau gunakan --pkl")
        path = Path(args.csv)
        print(f"sumber  : {path} (csv)")
        df = pd.read_csv(path, encoding="utf-8-sig")

    print(f"baris   : {len(df):,}")

    butuh = {"review_id", "teks_preprocessed", "sentimen"}
    kurang = butuh - set(df.columns)
    if kurang:
        print(
            "\nKolom kurang: " + ", ".join(sorted(kurang)) + "\n"
            "Berkas ini harus CSV hasil preprocessing (dengan kolom "
            "review_id, teks_preprocessed, sentimen) atau file .pkl label.\n"
            "CSV mentah (ulasan_wfc_*.csv) harus lewat tahap preprocessing dulu."
        )
        return 2

    start = time.time()
    metrics, model, vec, split = fit_evaluate_manual(df)
    print(f"elapsed : {round(time.time() - start, 1)} s")
    print(f"accuracy     : {round(metrics['accuracy'], 6)}")
    print(f"macro_f1     : {round(metrics['macro']['f1'], 6)}")
    print(f"weighted_f1  : {round(metrics['weighted']['f1'], 6)}")
    print(f"matrix       : {metrics['matrix']}")
    print(f"label train/test : {metrics['jumlah_label_training']:,} / "
          f"{metrics['jumlah_label_testing']:,}")
    print(f"review train/test: {metrics['jumlah_review_training']:,} / "
          f"{metrics['jumlah_review_testing']:,}")
    print(f"vocab tfidf      : {metrics['kosakata_tfidf']:,}")
    print(f"dihapus identik  : {metrics['baris_testing_dikeluarkan_teks_identik']:,}")
    print(f"kelas            : {metrics['classes']}")
    for name, row in metrics["per_class"].items():
        print(f"  {name}: " + ", ".join(f"{k}={round(v, 4)}" for k, v in row.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
