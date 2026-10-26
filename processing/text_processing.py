"""Pipeline pemrosesan teks Nafinder.

Aturan label dan enam tahap preprocessing disalin dari preprocessing.ipynb.
Sastrawi hanya dipakai untuk stemming; algoritma klasifikasi ditulis manual
pada modul naive_bayes.py.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Aturan pemotongan boilerplate Google Maps (urut dan pola mengikuti notebook).
PENANDA_TEMPLATE = re.compile(
    r"(?:Jenis pesanan|Jenis makanan|Jenis tempat|Tarif per orang|Waktu tunggu|"
    r"Makanan:\s*\d|Layanan:\s*\d|Suasana:\s*\d|Tingkat kebisingan|"
    r"Tempat parkir|Opsi tempat parkir|Jenis tempat duduk|Kesesuaian|Sumber daya|"
    r"Fasilitas untuk|Aksesibilitas|Layanan pelanggan|Pilihan makanan|Kategori)",
    flags=re.I,
)
PECAH_BAGIAN = re.compile(
    r",\s*|\s+(?:tapi|namun|tetapi|padahal|meski|walaupun|sedangkan|however|although)\s+",
    flags=re.I,
)
TAHAN_BERSIH = " ;:.!?\t\"'“”‘’"

KAMUS_NORMALISASI = {
    "gk":"tidak", "ga":"tidak", "gak":"tidak", "nggak":"tidak", "enggak":"tidak", "tdk":"tidak", "kagak":"tidak",
    "bgt":"sangat", "banget":"sangat", "bener":"benar", "yg":"yang", "dgn":"dengan", "utk":"untuk", "krn":"karena",
    "karna":"karena", "tp":"tapi", "aja":"saja", "jg":"juga", "udh":"sudah", "udah":"sudah", "sdh":"sudah", "blm":"belum",
    "belom":"belum", "dr":"dari", "sm":"sama", "sy":"saya", "klo":"kalau", "kalo":"kalau", "bwt":"buat", "bs":"bisa",
    "org":"orang", "bbrp":"beberapa", "gitu":"begitu", "gini":"begini", "makasih":"terima kasih", "trims":"terima kasih",
    "thx":"terima kasih", "recommended":"rekomendasi", "recomend":"rekomendasi", "worth it":"sepadan", "cozy":"nyaman",
    "resto":"restoran", "nget":"", "sih":"", "dong":"", "deh":"", "lah":"", "kah":"", "pun":"",
}
NEGASI_DIPERTAHANKAN = {"tidak","bukan","belum","jangan","tanpa","tak","tetapi","namun","tapi","meski","meskipun","walaupun"}
NEGASI = {"tidak","bukan","belum","jangan","tanpa","gak","gk","nggak","enggak","tdk","kurang","anti","ga","blm","blum","jgn","ngga","engga","gajadi","belom"}

POSITIF = {
    "enak","nikmat","mantap","bagus","nyaman","ramah","cepat","recommended","rekomendasi","suka","puas",
    "fresh","segar","worth","oke","ok","berkesan","rapi","bersih","murah","terjangkau","favorit","favorite",
    "great","good","delicious","nice","cozy","clean","helpful","friendly","affordable","amazing","perfect",
    "instagramable","love","best","satisfied","lezat","mantul","uenak","wenak","gurih","creamy","sedap","harum","wangi","worthit",
    "asri","teduh","sejuk","adem","luas","lega","lapang","homey","asik","seru","kece","keren","aesthetic","cantik","indah",
    "sigap","responsif","tanggap","bersahabat","lengkap","komplit","memadai","hemat","juara","menarik","menyenangkan","senang","bahagia","membantu","perhatian","terbaik",
}
NEGATIF = {
    "jelek","kecewa","lambat","mahal","ketus","kotor","bau","pahit","antri","antre","jorok","berisik","overpriced","expensive","bad","disappointed","dirty","slow","cold","stale","mengecewakan",
    "sembarangan","panas","sumpek","sesak","lama","ribet","susah","quite high","kemahalan","hambar","dingin","tidak ramah","not worth","poor","worst","percuma","jutek",
    "basi","gosong","amis","judes","lelet","lemot","kacau","buruk","kesal","sebal","benci","nyebelin","nyeselin","kesel","ngeselin","parahnya","enek","eneg",
}
TOLAK = {
    "parah", "sepi", "penuh", "padat", "bising", "sulit", "repot", "pengap", "gerah", "bocor", "rusak",
    "berantakan", "tawar", "garing", "pecah", "sopan", "peduli", "malang", "biasa", "cukup",
    "pelayanan", "pelayan", "pelayanannya", "duduk", "pesanan", "tempat",
}

KATA_ASPEK = {
    "makanan_minuman": ["kopi", "coffee", "makanan", "minuman", "menu", "rasa", "enak", "beans", "biji", "latte", "matcha", "cokelat", "croissant", "roti", "kue", "cake", "dessert", "susu", "milk", "teh", "tea", "jus", "juice", "nasi", "mie", "pasta", "pizza", "segar", "fresh", "manis", "pahit", "asin", "pedas", "gurih", "piccolo", "cappuccino", "espresso", "americano", "mocha", "croffle", "fried rice", "kopi susu", "iced coffee", "hot chocolate"],
    "pelayanan": ["pelayanan", "service", "staff", "petugas", "pegawai", "kasir", "ramah", "ketus", "sopan", "waiter", "waitress", "servis", "respon", "ramahnya", "orangnya", "kindness", "friendly", "antri", "antre", "cs", "dilayani", "pelanggan", "lambat", "slow", "jutek", "takeaway"],
    "harga": ["harga", "price", "murah", "mahal", "worth", "terjangkau", "budget", "bayar", "rupiah", "kemahalan", "price range", "affordable", "expensive", "overpriced", "cheap", "charge", "kisaran harga", "quite high", "rp", "idr", "cukup mahal", "terlalu mahal"],
    "suasana": ["suasana", "ambience", "atmosphere", "vibe", "nyaman", "tempat", "dekor", "konsep", "aesthetic", "musik", "ramai", "sepi", "crowd", "pemandangan", "view", "panorama", "crowded", "cozy", "hangout", "spot", "foto", "instagramable", "tenang", "berisik", "kursi", "meja", "interior", "privat"],
    "fasilitas_wfc": ["wifi", "internet", "stopkontak", "colokan", "charging", "plug", "laptop", "kerja", "work", "wfh", "wfc", "meeting", "24 jam", "buka 24", "parkir", "parking", "mushola", "toilet", "workspace", "power", "sinyal", "bekerja", "kerjaan"],
    "kebersihan": ["bersih", "kotor", "kebersihan", "debu", "higienis", "sanitasi", "clean", "dirty", "hygiene", "lantai", "meja kotor", "bau", "jorok", "rapi", "hand sanitizer"],
}


def potong_template(teks: Any) -> str:
    """Tahap 0 — potong boilerplate Google Maps dari ekor teks ulasan."""
    teks = re.sub(r"\s+", " ", str(teks)).strip()
    match = PENANDA_TEMPLATE.search(teks)
    return teks[:match.start()].strip() if match else teks


def pecah_subkalimat(teks: Any) -> list[str]:
    """Pecah satu ulasan menjadi sub-kalimat sesuai preprocessing.ipynb."""
    teks = re.sub(r"\s+", " ", str(teks)).strip()
    potongan = [p.strip(TAHAN_BERSIH) for p in PECAH_BAGIAN.split(teks)]
    return [p for p in potongan if p]


def deteksi_aspek(teks: Any) -> list[str]:
    """Deteksi seluruh aspek yang dibahas pada satu sub-kalimat sesuai preprocessing.ipynb."""
    t = f" {str(teks).lower()} "
    return [a for a, kata in KATA_ASPEK.items() if any(k.strip() in t for k in kata)]


def skor_sentimen(teks: Any) -> int:
    """Hitung skor sentimen lexicon sesuai preprocessing.ipynb."""
    tokens = re.findall(r"[a-zA-Z']+", str(teks).lower())
    skor = 0
    for i, tok in enumerate(tokens):
        if tok in TOLAK:
            continue
        neg = any(n in tokens[max(0, i - 3):i] for n in NEGASI)
        if tok in POSITIF:
            skor += -1 if neg else 1
        elif tok in NEGATIF:
            skor += 1 if neg else -1
    return skor


def label_sentimen(teks: Any, rating: Any) -> str:
    """Tentukan label sentimen: positif / netral / negatif.

    Skor lexicon yang tidak nol menang. Bila skor nol, rating bintang dipakai
    sebagai cadangan (rating 5 -> positif, rating 1 -> negatif, selain itu netral).

    Contoh:
        ("makanannya enak", 3)    -> "positif"
        ("lumasan biasa saja", 5) -> "positif"  (skor 0, rating 5)
        ("lumasan biasa saja", 3) -> "netral"
    """
    skor = skor_sentimen(teks)
    if skor > 0:
        return "positif"
    if skor < 0:
        return "negatif"
    try:
        if rating is not None and float(rating) >= 5:
            return "positif"
        if rating is not None and float(rating) <= 1:
            return "negatif"
    except (TypeError, ValueError):
        pass
    return "netral"


class TextPreprocessor:
    """Menjalankan enam tahap notebook; cache stem kata agar cepat."""

    def __init__(self):
        self.stemmer = StemmerFactory().create_stemmer()
        # ArrayDictionary Sastrawi default melakukan pencarian linear pada
        # daftar >30 ribu kata. Set lookup mempertahankan hasil stemming,
        # tetapi membuat pengecekan kamus O(1) sehingga proses dataset layak.
        dictionary = self.stemmer.delegatedStemmer.dictionary
        if hasattr(dictionary, "words"):
            dictionary.words = set(dictionary.words)
        self.stopwords = set(StopWordRemoverFactory().get_stop_words())
        self.stopwords |= {"nih", "yuk", "kok", "loh", "hehe", "wkwk", "nya"}
        self.stopwords -= NEGASI_DIPERTAHANKAN

    @staticmethod
    def tahap1_cleaning(teks: Any) -> str:
        """Tahap 1 — Cleaning: buang URL, mention, hashtag, emoji, dan tanda baca.

        Menghapus: http/www, @mention, #hashtag, emoji, dan seluruh karakter
        non-huruf. Spasi berlebih dirapikan menjadi satu spasi.

        Contoh:
            "Kopi enak @user #viral http://x.co" -> "Kopi enak"
        """
        teks = re.sub(r"http\S+|www\S+", " ", str(teks))
        teks = re.sub(r"@\w+|#\w+", " ", teks)
        teks = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", " ", teks)
        teks = re.sub(r"[^a-zA-Z\s]", " ", teks)
        return re.sub(r"\s+", " ", teks).strip()

    @staticmethod
    def tahap2_case_folding(teks: str) -> str:
        """Tahap 2 — Case Folding: ubah seluruh huruf menjadi huruf kecil.

        Menyamakan representasi kata agar "Kopi" dan "kopi" terhitung sama.

        Contoh:
            "Kopi Enak" -> "kopi enak"
        """
        return teks.lower()

    @staticmethod
    def tahap3_tokenizing(teks: str) -> list[str]:
        """Tahap 3 — Tokenizing: pecah teks menjadi daftar kata.

        Pemecah pada spasi (hasil tahap 1 sudah bebas tanda baca).

        Contoh:
            "kopi enak banget" -> ["kopi", "enak", "banget"]
        """
        return teks.split()

    @staticmethod
    def tahap4_normalisasi(tokens: list[str]) -> list[str]:
        """Tahap 4 — Normalization: ganti kata tidak baku ke bentuk baku.

        Memakai KAMUS_NORMALISASI (mis. "gk"->"tidak", "bgt"->"sangat",
        "yg"->"yang", "makasih"->"terima kasih"). Kata tak dikenal dibiarkan.

        Contoh:
            ["makanannya", "gk", "enak"] -> ["makanannya", "tidak", "enak"]
        """
        return [KAMUS_NORMALISASI.get(t, t) for t in tokens]

    def tahap5_stopword(self, tokens: list[str]) -> list[str]:
        """Tahap 5 — Stopword Removal: buang kata umum tanpa makna sentimen.

        Memakai daftar stopword Sastrawi ditambah kata sapaan ("nih", "yuk",
        "kok", "loh", "hehe", "wkwk", "nya").

        PENTING: kata negasi ("tidak", "bukan", "belum", ...) dikecualikan dari
        daftar stopword, karena menghapus "tidak" akan membalik makna sentimen
        ("tidak enak" menjadi "enak").

        Contoh:
            ["kopi", "saya", "tidak", "enak"] -> ["kopi", "tidak", "enak"]
        """
        return [t for t in tokens if t and t not in self.stopwords]

    @lru_cache(maxsize=100_000)
    def stem(self, token: str) -> str:
        """Stemming satu kata dengan Sastrawi (satu-satunya pakai pustaka pihak ketiga).

        Cache LRU 100.000 entri karena stemming kata yang sama berulang kali
        merupakan sebagian besar waktu proses pada dataset besar.
        """
        return self.stemmer.stem(token)

    def tahap6_stemming(self, tokens: list[str]) -> list[str]:
        """Tahap 6 — Stemming: stemmed setiap token ke bentuk akar katanya.

        Contoh:
            ["makanannya", "enak", "sekali"] -> ["makan", "enak", "sekali"]
        """
        return [self.stem(t) for t in tokens]

    def preprocess(self, teks: Any) -> str:
        """Jalankan enam tahap berurutan dan kembalikan teks siap vektorisasi.

        Urutan: cleaning -> case folding -> tokenizing -> normalisasi ->
        stopword -> stemming. Kata hasil stemming dengan panjang < 3 huruf
        dibuang (noise seperti "yg", "di" yang tersisa).

        Contoh:
            "Kopi Enak Banget, Gak recommended!" -> "kopi enak banget rekomendasikan"
        """
        t = self.tahap2_case_folding(self.tahap1_cleaning(teks))
        tokens = self.tahap5_stopword(self.tahap4_normalisasi(self.tahap3_tokenizing(t)))
        return " ".join(w for w in self.tahap6_stemming(tokens) if len(w) >= 3)

    def semua_tahap(self, teks: Any) -> list[dict[str, Any]]:
        """Jalankan keenam tahap dan kembalikan hasil tiap tahap (untuk Explainable AI).

        Dipakai untuk menampilkan tabel "sebelum–sesudah" tiap tahap preprocessing
        pada antarmuka atau lampiran skripsi.

        Returns:
            List dict dengan kunci: tahap, hasil (teks/daftar token), jumlah_kata.
        """
        t1 = self.tahap1_cleaning(teks)
        t2 = self.tahap2_case_folding(t1)
        t3 = self.tahap3_tokenizing(t2)
        t4 = self.tahap4_normalisasi(t3)
        t5 = self.tahap5_stopword(t4)
        t6 = [w for w in self.tahap6_stemming(t5) if len(w) >= 3]
        return [
            {"tahap": "1. Cleaning", "hasil": t1, "jumlah_kata": len(t1.split())},
            {"tahap": "2. Case Folding", "hasil": t2, "jumlah_kata": len(t2.split())},
            {"tahap": "3. Tokenizing", "hasil": " | ".join(t3), "jumlah_kata": len(t3)},
            {"tahap": "4. Normalization", "hasil": " | ".join(t4), "jumlah_kata": len(t4)},
            {"tahap": "5. Stopword Removal", "hasil": " | ".join(t5), "jumlah_kata": len(t5)},
            {"tahap": "6. Stemming", "hasil": " | ".join(t6), "jumlah_kata": len(t6)},
        ]


def proses_ulasan(teks: Any, rating: Any, preprocessor: TextPreprocessor) -> list[dict[str, Any]]:
    """Ulasan mentah -> potong template -> sub-kalimat -> aspek/sentimen -> preprocessing."""
    teks_bersih = potong_template(teks)
    hasil = []
    for sub in pecah_subkalimat(teks_bersih):
        aspek_list = deteksi_aspek(sub)
        if not aspek_list:
            continue
        processed = preprocessor.preprocess(sub)
        if not processed:
            continue
        sentimen = label_sentimen(sub, rating)
        for aspek in aspek_list:
            hasil.append({
                "sub_kalimat": sub,
                "teks_preprocessed": processed,
                "aspek": aspek,
                "sentimen": sentimen,
            })
    return hasil
