#!/usr/bin/env python3
# ===================================================================
# SCRAPING ULASAN KAFE WFC GOOGLE MAPS
#
# Nama    : Fadhli Ilham Nafi'an Yuswono (23.18.001)
# Jurusan : Teknik Informatika S-1 - Institut Teknologi Nasional Malang
# Skripsi : Nafinder - Rekomendasi Kafe WFC Kota Malang dengan
#           Multinomial Naive Bayes
#
# Metode  : Selenium (tanpa API berbayar)
# Input   : daftar_kafe_final.csv
# Output  : ulasan_wfc_final.csv
#
# Tahapan: baca daftar kafe -> buka tab Ulasan per kafe -> scroll &
#          baca kartu ulasan -> checkpoint CSV per 100 ulasan ->
#          hitung kolom valid.
# ===================================================================
"""
Cara pakai
----------
Jalankan seluruh daftar kafe:
    python scrapping_gmaps_nafii.py

Tes singkat, 1 kafe dan 20 ulasan:
    python scrapping_gmaps_nafii.py --maks-kafe 1 --max 20

Tab Ulasan tidak muncul karena Chrome belum login? Buka Chrome dengan mode
debug, login sekali, lalu jalankan dengan --attach:
    chrome.exe --remote-debugging-port=9222 --user-data-dir="D:\\chrome_profile"
    python scrapping_gmaps_nafii.py --attach

Lanjutkan setelah sempat berhenti (progress tersimpan otomatis):
    python scrapping_gmaps_nafii.py --mulai-dari 20

Hitung ulang kolom valid setelah seluruh proses selesai:
    python scrapping_gmaps_nafii.py --finalisasi

Catatan:
  - Google Maps me-render ulasan bertahap saat scroll, jadi target per kafe
    adalah batas atas, bukan jaminan.
  - Google kadang hanya menampilkan tanggal relatif seperti "3 bulan lalu",
    jadi kolom tanggal diisi tanggal absolut lebih dulu lalu yang relatif.
  - Script tidak memakai API berbayar, tapi bisa kena CAPTCHA. Bila kena,
    script berhenti rapi dan bisa dilanjutkan dengan --mulai-dari.
"""


import argparse
import csv
import json
import os
import random
import re
import sys
import time
from urllib.parse import quote_plus

from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait


# ===================================================================
# KONFIGURASI
# ===================================================================
FILE_KAFE = "daftar_kafe_final.csv"      # daftar kafe hasil pencarian
FILE_HASIL = "ulasan_wfc_final.csv"      # hasil akhir scraping
FILE_PROGRESS = "progress_wfc_final.json"

TARGET_PER_KAFE = 2000   # batas atas per kafe; berhenti juga kalau data habis
MAKS_KEKOSONGAN = 8      # berapa kali scroll tanpa ulasan baru sebelum nyerah
JEDA_SCROLL = (1.0, 2.0)  # jeda acak tiap scroll
JEDA_ANTAR_KAFE = (8.0, 16.0)

KOLOM = [  # kolom yang dipakai notebook preprocessing
    "placeId", "nama_kafe", "teks_ulasan", "rating_ulasan",
    "tanggal_ulasan", "kafe_id", "valid",
]


# ===================================================================
# UTILITAS (pencarian, verifikasi, pembacaan halaman)
# ===================================================================

def port_terbuka(port):
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.2)
    ok = s.connect_ex(("127.0.0.1", port)) == 0
    s.close()
    return ok

def bikin_driver(attach=False, port=9222):
    if attach:
        # Tempel ke Chrome yang sudah dibuka dengan --remote-debugging-port
        if not port_terbuka(port):
            sys.exit(
                f"Port {port} tidak terbuka -> Chrome dengan flag debug BELUM dijalankan.\n"
                "Jalankan dulu di PowerShell TERPISAH (jangan ditutup selagi scraping):\n"
                f'  & "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" '
                f"--remote-debugging-port={port} "
                '--user-data-dir="D:\\Skripsi\\Scraping Local\\chrome_profile"\n'
                "Lalu login Google di jendela itu (sekali saja), biarkan Chrome terbuka,\n"
                "dan jalankan lagi perintah ini."
            )
        o = Options()
        o.debugger_address = f"127.0.0.1:{port}"
        return webdriver.Chrome(options=o)
    o = Options()
    o.add_argument("--lang=id")
    o.add_argument("--accept-language=id-ID,id")
    o.add_argument("--log-level=3")
    o.add_argument("--disable-notifications")
    o.add_argument("--disable-blink-features=AutomationControlled")
    o.add_experimental_option("excludeSwitches", ["enable-automation"])
    o.add_experimental_option("useAutomationExtension", False)
    # Tanpa --user-data-dir, memakai profil Chrome default.
    return webdriver.Chrome(options=o)

def url_pencarian(nama, place_id=""):
    """URL pencarian resmi Google Maps, bukan URL /place/ yang disusun manual.

    Link mentah kadang rusak seperti /place//@ sehingga halaman kosong.
    URL /maps/search/?api=1&query=... selalu di-encode, sehingga nama kafe
    yang mengandung "/" atau "&" tidak merusak path, dan query_place_id
    memastikan cabang yang dibuka adalah yang benar.
    """
    url = ("https://www.google.com/maps/search/?api=1"
           f"&query={quote_plus(nama)}&hl=id")
    if place_id:
        url += f"&query_place_id={place_id}"
    return url

def place_id_dari_url(url: str) -> str:
    m = re.search(r"(ChIJ[A-Za-z0-9_\-]{10,})", url or "")
    return m.group(1) if m else ""

def deteksi_kebijakan(driver) -> bool:
    """True kalau kena CAPTCHA / halaman 'belum memenuhi syarat'."""
    u = (driver.current_url or "").lower()
    if "sorry" in u or "interstitial" in u:
        return True
    try:
        t = (driver.find_element(By.TAG_NAME, "body").text or "").lower()[:800]
    except Exception:
        return False
    return any(k in t for k in ("tidak memenuhi syarat", "belum memenuhi syarat",
                                "sistem mendeteksi aktivitas", "unusual traffic"))

def baca_alamat(driver) -> str:
    """Ambil alamat dari halaman kafe untuk filter Kota Malang dan untuk
    membangun kafe_id berformat 'Nama Kafe | Jalan'."""
    for sel in ('a[data-item-id="address"] .Io6YTe',
                'button[data-item-id="address"] .Io6YTe',
                'button[data-item-id="address"]',
                'div.Io6YTe[data-value]'):
        try:
            t = (driver.find_element(By.CSS_SELECTOR, sel).text or "").strip()
            if t:
                return t
        except Exception:
            continue
    # fallback dari aria-label tombol alamat
    try:
        for b in driver.find_elements(By.CSS_SELECTOR, "button.CsEn2e, a.CsEn2e"):
            aria = (b.get_attribute("aria-label") or "")
            if "alamat" in aria.lower() or "address" in aria.lower():
                t = aria.split(":", 1)[-1].strip()
                if t:
                    return t
    except Exception:
        pass
    return ""

def aman_teks(driver, css, fallback=""):
    try:
        el = driver.find_element(By.CSS_SELECTOR, css)
        return (el.text or el.get_attribute("aria-label") or "").strip()
    except Exception:
        return fallback

def di_kota_malang(alamat: str) -> bool:
    """Khusus KOTA Malang — Kabupaten Malang harus terbuang.
    Beberapa alamat Google ditulis dalam bahasa Inggris, jadi polanya dua-duanya."""
    a = (alamat or "").lower()
    if not a:
        return False
    if "kabupaten malang" in a:
        return False
    return ("kota malang" in a) or ("malang city" in a)

def kafe_id_dari(nama: str, alamat: str) -> str:
    """Format 'Nama Kafe | Jalan', dengan nama kafe lalu jalan sebagai
    penanda agar setiap kafe memiliki kafe_id sendiri."""
    jalan = (alamat.split(",")[0] or "").strip()
    return f"{nama.strip()} | {jalan}"

def jumlah_ulasan(driver):
    """Hitung kartu ulasan yang sudah dirender pada div[data-review-id]."""
    try:
        return len(driver.find_elements(By.CSS_SELECTOR, 'div[data-review-id]'))
    except Exception:
        return 0

def url_bersih(url: str) -> str:
    """Buang seluruh query string dari link Google Maps.

    Link dengan query seperti ?authuser=0&hl=en&rclk=1 membuat Maps membuka
    versi Inggris sehingga tab Ulasan tidak tersedia, sedangkan URL tanpa
    query menampilkan tab [Ringkasan, Menu, ULASAN, Tentang].
    """
    return str(url).split("?")[0]

def nama_cocok(nama_csv, h1):
    """Verifikasi ringan: nama CSV dan h1 saling mengandung, tanpa memperhatikan
    huruf besar-kecil, spasi, dan tanda baca."""
    if not nama_csv or not h1:
        return False
    a = re.sub(r"[^a-z0-9]+", "", nama_csv.lower())
    b = re.sub(r"[^a-z0-9]+", "", h1.lower())
    return a in b or b in a

def simpan_debug(driver, nama):
    """Simpan bukti kalau verifikasi/ulasan gagal -> debug_<nama>.png/.html."""
    nm = re.sub(r"[^A-Za-z0-9]+", "_", (nama or "tanpa"))[:40]
    try:
        driver.save_screenshot(f"debug_{nm}.png")
        with open(f"debug_{nm}.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print(f"    debug tersimpan: debug_{nm}.png / debug_{nm}.html")
    except Exception as e:
        print(f"    debug gagal: {str(e)[:80]}")


# ===================================================================
# KLIK & SCROLL TAB ULASAN
# ===================================================================

def klik(driver, el):
    """Klik tab ulasan sampai tab benar-benar berpindah.

    Klik sampai ke DOM belum tentu sampai ke handler React Google Maps, jadi
    dicoba berurutan dari yang paling menyerupai klik manusia:
      1. scrollIntoView lalu klik Selenium
      2. ActionChains berbasis koordinat elemen
      3. MouseEvent bubbles=true karena React memakai event delegation
      4. executeScript click sebagai cadangan terakhir
    Setiap langkah ditunggu sejenak lalu dicek apakah tab sudah berpindah.
    """
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.common.exceptions import (
        ElementClickInterceptedException, ElementNotInteractableException,
        WebDriverException)
    import time as _t

    def pindah():
        """True kalau tab Reviews yang aktif / ulasannya bertambah."""
        try:
            for t in driver.find_elements(By.CSS_SELECTOR, 'div[role="tablist"] button'):
                s = ((t.get_attribute("aria-label") or "") + " " + (t.text or "")).lower()
                if any(k in s for k in ("ulasan", "review")):
                    if (t.get_attribute("aria-selected") == "true"
                            or t.get_attribute("aria-pressed") == "true"
                            or (t.get_attribute("class") or "").find("selected") >= 0):
                        return True
        except Exception:
            pass
        return jumlah_ulasan(driver) > 0

    # 1) scroll + native
    try:
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
        _t.sleep(0.4)
        el.click()
        _t.sleep(1.2)
        if pindah():
            return "native"
    except (ElementClickInterceptedException, ElementNotInteractableException, WebDriverException):
        pass

    # 2) ActionChains: klik di titik koordinat sebenarnya
    try:
        ActionChains(driver).move_to_element(el).pause(0.3).click().perform()
        _t.sleep(1.2)
        if pindah():
            return "actionchains"
    except Exception:
        pass

    # 3) MouseEvent dengan bubbles (React pakai event delegation)
    for opts in ({"bubbles": True, "cancelable": True},
                 {"bubbles": True, "cancelable": True, "composed": True}):
        try:
            driver.execute_script(
                "arguments[0].dispatchEvent(new MouseEvent('click', "
                "arguments[1]));", el, opts)
            _t.sleep(1.2)
            if pindah():
                return "mouseevent"
        except Exception:
            continue

    # 4) cadangan terakhir
    try:
        driver.execute_script("arguments[0].click();", el)
        _t.sleep(1.2)
        if pindah():
            return "script-click"
    except Exception:
        pass

    return "gagal"

def tab_ulasan_aktif(driver):
    """True/False kalau tab Ulasan terdeteksi aktif; None kalau tidak tahu."""
    try:
        for t in driver.find_elements(By.CSS_SELECTOR, 'div[role="tablist"] button'):
            s = ((t.get_attribute("aria-label") or "") + " " + (t.text or "")).lower()
            if "ulasan" in s or "review" in s:
                if (t.get_attribute("aria-selected") == "true"
                        or t.get_attribute("aria-pressed") == "true"
                        or "selected" in (t.get_attribute("class") or "").lower()):
                    return True
                return False
    except Exception:
        pass
    return None

def daftar_terbuka(driver, n, dasar=0):
    """True bila daftar ulasan terbuka, bukan sekadar preview Ringkasan."""
    if n <= 0:
        return False
    if n > dasar:
        return True
    return tab_ulasan_aktif(driver) is not False

def tunggu_daftar_ulasan(driver, detik=20, dasar=0):
    """Tunggu kartu ulasan setelah tab dipilih; scroll hanya panel ulasan."""
    batas = time.time() + detik
    while time.time() < batas:
        n = jumlah_ulasan(driver)
        if daftar_terbuka(driver, n, dasar):
            return n
        # Jangan scroll ancestor umum karena bisa menggeser hasil pencarian
        # dan peta; tab Ulasan memakai panelnya sendiri.
        jalur = scroll_panel_ulasan(driver)
        if jalur == "tidak-ada":
            time.sleep(1)
    return jumlah_ulasan(driver)

def scroll_panel_ulasan(driver):
    """Geser hanya panel ulasan, bukan seluruh sidebar.

    Jangan scroll div[role="main"] atau ancestor tablist generik karena
    keduanya mencakup pane kiri lengkap dengan riwayat pencarian, sehingga
    peta dan sidebar ikut bergeser. Target berurutan:
      1. ancestor scrollable dari kartu ulasan bila sudah ada
      2. wadah konten tab .DxyBCb
      3. ancestor scrollable dari tombol urutkan atau saring ulasan
    Mengembalikan penanda jalur yang berhasil.
    """
    jalur = driver.execute_script(r"""
      // 1) kartu sudah ada -> wadahnya
      const rev = document.querySelector('div[data-review-id]');
      if (rev) {
        let el = rev.parentElement;
        while (el && el !== document.body) {
          if (el.scrollHeight > el.clientHeight + 40) {
            el.scrollTop = el.scrollHeight; return 'kartu';
          }
          el = el.parentElement;
        }
      }
      // 2) panel konten tab (unik, bukan sidebar utama)
      for (const e of document.querySelectorAll('div[class*="DxyBCb"]')) {
        if (e.scrollHeight > e.clientHeight + 40) {
          e.scrollTop = e.scrollHeight; return 'panel-tab';
        }
      }
      // 3) naik dari tombol filter ulasan
      for (const b of document.querySelectorAll('button[aria-label]')) {
        const a = b.getAttribute('aria-label') || '';
        if (!/Urutkan ulasan|Saring ulasan|Telusuri ulasan/i.test(a)) continue;
        let p = b.parentElement;
        while (p && p !== document.body) {
          if (p.scrollHeight > p.clientHeight + 40) {
            p.scrollTop = p.scrollHeight; return 'dari-filter';
          }
          p = p.parentElement;
        }
      }
      return 'tidak-ada';
    """)
    if jalur != "tidak-ada":
        time.sleep(random.uniform(*JEDA_SCROLL))
    else:
        time.sleep(1)
    return jalur


# ===================================================================
# BACA KARTU ULASAN (satu executeScript untuk semua kartu)
# ===================================================================

def baca_ulasan(driver, expand=True):
    """Baca seluruh kartu ulasan yang sudah terender dalam satu panggilan JS.

    Satu executeScript membaca semua kartu sekaligus dan mengklik seluruh
    tombol "Lainnya" serentak, sehingga tidak ada round-trip Selenium per
    kartu maupun jeda per kartu. Mengembalikan list of dict.
    """
    if expand:
        # buka semua teks panjang serentak; tombol hilang setelah diklik,
        # jadi klik ulang pada kartu yang sudah kebuka tidak akan terjadi
        try:
            driver.execute_script("""
              for (const b of document.querySelectorAll('button.w8nwRe')) {
                try { b.click(); } catch (e) {}
              }
            """)
        except Exception:
            pass

    try:
        data = driver.execute_script(r"""
      const out = [];
      for (const c of document.querySelectorAll('div[data-review-id]')) {
        const rid = c.getAttribute('data-review-id') || '';
        const q = sel => {
          const e = c.querySelector(sel);
          return e ? e.textContent.trim() : '';
        };

        // rating: aria-label "5 bintang" / "Rated 5" -> angka 1-5
        let rating = '';
        const r1 = c.querySelector('span.kvMYJc');
        const r2 = c.querySelector('div[role="img"][aria-label]');
        const lab = ((r1 && r1.getAttribute('aria-label')) ||
                     (r2 && r2.getAttribute('aria-label')) || '');
        const m = lab.match(/[1-5]/);
        if (m) rating = m[0];

        // teks ulasan (selector lama & baru), spasi dirapikan
        let teks = q('div.MyEned') || q('div.wiI7pd') || q('span[class*="MyEned"]');
        teks = teks.replace(/\s+/g, ' ').trim();

        // tanggal: absolut (ada tahun) lebih diprioritaskan dari relatif
        let tgl_abs = '', tgl_rel = '';
        for (const s of c.querySelectorAll('span.rsqaWe, span[aria-label], span[title]')) {
          for (const raw of [s.getAttribute('aria-label'),
                             s.getAttribute('title'), s.textContent]) {
            if (!raw) continue;
            const t = raw.trim();
            if (!tgl_abs && /\b(19|20)\d{2}\b/.test(t)) tgl_abs = t;
            else if (!tgl_rel &&
                     /\b(lalu|hari|minggu|bulan|tahun|kemarin)\b/i.test(t)) tgl_rel = t;
          }
          if (tgl_abs) break;
        }

        out.push({id: rid, rating: rating, teks: teks,
                  tanggal_abs: tgl_abs, tanggal_rel: tgl_rel});
      }
      return out;
    """)
    except Exception as e:
        # jaringan lambat atau re-render halaman bisa membuat eksekusi JS
        # gagal; coba sekali lagi setelah jeda, dan bila tetap gagal putaran
        # ini dianggap kosong agar loop mencobanya lagi pada putaran berikutnya.
        print(f"    ! baca DOM gagal ({type(e).__name__}), coba lagi dalam 3s")
        time.sleep(3)
        try:
            data = driver.execute_script(r"""
              const out = [];
              for (const c of document.querySelectorAll('div[data-review-id]')) {
                const q = sel => {
                  const e = c.querySelector(sel);
                  return e ? e.textContent.trim() : '';
                };
                const r = c.querySelector('span.kvMYJc')
                       || c.querySelector('div[role="img"][aria-label]');
                const m = ((r && r.getAttribute('aria-label')) || '').match(/[1-5]/);
                out.push({id: c.getAttribute('data-review-id') || '',
                          rating: m ? m[0] : '',
                          teks: (q('div.MyEned') || q('div.wiI7pd')).replace(/\s+/g,' ').trim(),
                          tanggal_abs: '', tanggal_rel: ''});
              }
              return out;
            """)
            return data or []
        except Exception:
            return []
    return data or []


# ===================================================================
# BUKA KAFE & PINDAH KE TAB ULASAN
# ===================================================================

def buka_tempat_dari_cari(driver, wait, nama, place_id=""):
    """Buka kafe lewat URL pencarian resmi lalu verifikasi namanya.

    Mengembalikan (h1_di_halaman, url_akhir); pemanggil yang memutuskan
    cocok atau tidak. Bila Maps membuka daftar hasil, klik hasil pertama.
    """
    driver.get(url_pencarian(nama, place_id))
    time.sleep(random.uniform(3.0, 5.0))

    # Chrome merender lebih lambat bila tab bukan tab aktif, sehingga kafe
    # bisa terlewat gagal verifikasi. Beri peringatan, jangan diam saja.
    try:
        if driver.execute_script("return document.hidden;"):
            print("    ! PERINGATAN: tab Maps sedang di BELAKANG (bukan tab aktif).")
            print("      Render bisa tersendat -> kafe bisa terlewat gagal verifikasi.")
            print("      Biarkan tab Maps kelihatan / pakai browser lain untuk chat.")
    except Exception:
        pass

    # kadang muncul daftar hasil pencarian (beberapa cabang) -> klik yang pertama
    try:
        for _ in range(8):
            h1 = aman_teks(driver, "h1.DUwDvf") or aman_teks(driver, "h1")
            if h1 and "/place/" in (driver.current_url or ""):
                break
            hasil = driver.find_elements("css selector", "a.hfpxzc")
            if hasil:
                driver.execute_script("arguments[0].click();", hasil[0])
                time.sleep(4)
            else:
                time.sleep(1)
    except Exception:
        pass

    # h1 kadang baru muncul beberapa saat setelah halaman termuat; polling 15 detik.
    h1 = aman_teks(driver, "h1.DUwDvf") or aman_teks(driver, "h1")
    for _ in range(15):
        if h1:
            break
        time.sleep(1)
        h1 = aman_teks(driver, "h1.DUwDvf") or aman_teks(driver, "h1")
    return h1, driver.current_url

def buka_tab_ulasan(driver, wait, url, nama_csv="", place_id=""):
    """Buka kafe lalu berpindah ke tab Ulasan.

    Navigasi memakai URL pencarian resmi (nama + place_id) karena link mentah
    dari CSV bisa rusak seperti /place//@ atau menunjuk cabang yang salah.

    Mengembalikan:
      (nama, alamat)  -> sukses
      False           -> gagal tab, salah nama, atau di luar Kota Malang
      None            -> kena CAPTCHA sehingga proses berhenti total
    """
    # --- navigasi via URL pencarian resmi ---
    if place_id or nama_csv:
        h1_hal, url_akhir = buka_tempat_dari_cari(
            driver, wait, nama_csv or url, place_id)
        if not nama_cocok(nama_csv, h1_hal):
            print(f"    ! VERIFIKASI GAGAL: CSV={nama_csv!r} vs halaman={h1_hal!r}")
            print("      -> kafe dilewati (kemungkinan salah cabang)")
            simpan_debug(driver, nama_csv)
            return False
        print(f"    [cari] OK: {h1_hal!r} | {url_akhir[:90]}")
    else:
        # cadangan: tanpa nama/place_id, buang query string saja
        url_buka = url_bersih(url)
        driver.get(url_buka)
        time.sleep(random.uniform(3.0, 5.0))

    if deteksi_kebijakan(driver):
        print("    !! KENAK CAPTCHA/KEBIJAKAN. Script berhenti total.")
        return None

    nama = aman_teks(driver, "h1.DUwDvf") or aman_teks(driver, "h1")
    alamat = baca_alamat(driver)

    # ---- filter Kota Malang ----
    if alamat and not di_kota_malang(alamat):
        print(f"    dilewati, di luar Kota Malang: {alamat[:70]}")
        return False

    dasar = jumlah_ulasan(driver)   # bisa sudah >0 (preview di tab Overview)
    # URL disimpan setelah h1 lolos verifikasi di atas
    url_setelah_buka = driver.current_url

    def _dasar_place(u):
        """Bandingkan hanya bagian /place/Nama — buang /data= dan /@..."""
        u = (u or "").split("/data=")[0].split("?")[0]
        i = u.find("/@")
        return u[:i] if i > 0 else u

    def masih_di_tempat(nama_yg_dimaksud):
        """True bila masih berada di kafe yang sama.

        h1 bisa kosong sementara saat tab Ulasan dirender ulang, jadi ada
        cadangan berupa pembandingan bagian /place/ pada URL terhadap URL
        saat verifikasi.
        """
        h1 = aman_teks(driver, "h1.DUwDvf") or aman_teks(driver, "h1")
        if h1:
            return nama_cocok(nama_yg_dimaksud, h1)
        u = driver.current_url or ""
        return "/place/" in u and _dasar_place(u) == _dasar_place(url_setelah_buka)

    # Strategi A: klik tab berdasarkan TEKS
    try:
        tabs = driver.find_elements(By.CSS_SELECTOR, 'div[role="tablist"] button')
        idx = None
        for i, t in enumerate(tabs):
            s = ((t.get_attribute("aria-label") or "") + " " + (t.text or "")).lower()
            if any(k in s for k in ("ulasan", "review")):
                idx = i
                break
        if idx is not None:
            mode = klik(driver, tabs[idx])
            n = tunggu_daftar_ulasan(driver, detik=20, dasar=dasar)
            print(f"      [A] tab[{idx}] klik={mode} ulasan {dasar}->{n}")
            if daftar_terbuka(driver, n, dasar) and masih_di_tempat(nama):
                return (nama, alamat)
        elif len(tabs) >= 2:
            mode = klik(driver, tabs[1])
            n = tunggu_daftar_ulasan(driver, detik=20, dasar=dasar)
            print(f"      [A] tab[1] cadangan klik={mode} ulasan {dasar}->{n}")
            if daftar_terbuka(driver, n, dasar) and masih_di_tempat(nama):
                return (nama, alamat)
        else:
            print("      [A] tidak ada tab")
    except Exception as e:
        print(f"      [A] gagal: {str(e)[:90]}")

    # Strategi B: klik tombol "N ulasan", dibatasi pada tablist dan pesan
    # tempat. Pencarian lewat xpath global pernah menangkap riwayat pencarian
    # di sidebar sehingga Maps berpindah tempat.
    try:
        kandidat = driver.find_elements(
            By.CSS_SELECTOR,
            'div[role="tablist"] button, div[role="tablist"] [role="button"], '
            'div[role="main"] button')
        klik_dicoba = 0
        for el in kandidat:
            try:
                s = ((el.get_attribute("aria-label") or "") + " " +
                     (el.text or "")).lower()
                if not re.search(r"\d\s*(ulasan|review)", s):
                    continue
                if not el.is_displayed():
                    continue
                klik_dicoba += 1
                klik(driver, el)
                n = tunggu_daftar_ulasan(driver, detik=15, dasar=dasar)
                if daftar_terbuka(driver, n, dasar) and masih_di_tempat(nama):
                    print(f"      [B] ok ulasan {dasar}->{n}")
                    alamat = alamat or baca_alamat(driver)
                    return (nama, alamat)
                if not masih_di_tempat(nama):
                    # tombolnya ternyata milik tempat lain -> buka ulang kafenya
                    print("      [B] halaman pindah tempat -> buka ulang")
                    h1_hal, _ = buka_tempat_dari_cari(
                        driver, wait, nama, place_id)
                    if not nama_cocok(nama, h1_hal):
                        print(f"      [B] buka ulang gagal ({h1_hal!r}) -> berhenti B")
                        break
                    dasar = jumlah_ulasan(driver)
                    # aktifkan lagi tab Ulasan
                    tabs = driver.find_elements(
                        By.CSS_SELECTOR, 'div[role="tablist"] button')
                    for t in tabs:
                        s2 = ((t.get_attribute("aria-label") or "") +
                              " " + (t.text or "")).lower()
                        if "ulasan" in s2 or "review" in s2:
                            klik(driver, t)
                            tunggu_daftar_ulasan(driver, detik=15, dasar=dasar)
                            break
            except Exception:
                continue
        if klik_dicoba == 0:
            print("      [B] tidak ada tombol 'N ulasan' di tab/pesan tempat")
    except Exception as e:
        print(f"      [B] gagal: {str(e)[:90]}")

    # Strategi C: scroll panel ulasan saja (bukan sidebar utama)
    jalur_scroll = scroll_panel_ulasan(driver)
    print(f"      [C] scroll awal: {jalur_scroll}")
    time.sleep(2)
    n = jumlah_ulasan(driver)
    if daftar_terbuka(driver, n, dasar) and masih_di_tempat(nama):
        print(f"      [C] ok setelah scroll {dasar}->{n}")
        alamat = alamat or baca_alamat(driver)
        return (nama, alamat)

    # gagal: simpan tangkapan layar dan file HTML untuk audit selector
    try:
        nm = re.sub(r"[^A-Za-z0-9]+", "_", (nama or "tanpa"))[:40]
        path = f"gagal_tab_{nm}.png"
        driver.save_screenshot(path)
        with open(f"gagal_tab_{nm}.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print(f"    ! tab ulasan gagal (ulasan {dasar}->{jumlah_ulasan(driver)}) -> {path}")
        print(f"      judul: {(driver.title or '')[:70]}")
    except Exception as e:
        print(f"    ! tab ulasan gagal, screenshot pun gagal: {str(e)[:70]}")
    return False


# ===================================================================
# LOOP UTAMA, CHECKPOINT, DAN FINALISASI
# ===================================================================

def kumpul_ulasan(driver, nama_kafe, place_id, kafe_id,
                  target=TARGET_PER_KAFE, simpan_batch=None):
    """Scroll sampai `target` ulasan atau data berhenti bertambah.

    simpan_batch(rows): opsional, dipanggil tiap kelipatan 100 ulasan dengan
    seluruh row yang sudah terkumpul. Pemanggil menghitung offsetnya sendiri,
    sehingga tidak ada baris ganda.
    """
    kumpul, kekosongan = {}, 0
    detik_mulai = time.time()
    t_sabit = 0   # detik terakhir baris status dicetak
    berikut_batch = 100   # checkpoint pertama di 100 ulasan

    while len(kumpul) < target and kekosongan < MAKS_KEKOSONGAN:
        sebelum = len(kumpul)
        for r in baca_ulasan(driver):
            if not r["teks"]:
                continue
            kunci = r["id"] or r["teks"][:80]
            if kunci not in kumpul:
                kumpul[kunci] = r

        if len(kumpul) == sebelum:
            kekosongan += 1
        else:
            kekosongan = 0

        # status dicetak tiap sekitar 4 detik agar terminal tidak membanjiri
        # scrollback, tetap terlihat prosesnya berjalan
        lewat = int(time.time() - detik_mulai)
        if lewat - t_sabit >= 4 or len(kumpul) >= target:
            t_sabit = lewat
            print(f"    [jalan] {len(kumpul):>4}/{target} ulasan "
                  f"| kekosongan {kekosongan}/{MAKS_KEKOSONGAN} "
                  f"| {lewat}s", flush=True)

        # checkpoint per 100 ulasan agar data sudah masuk file selama proses
        # berjalan; bila script mati, yang hilang hanya sisa batch
        if simpan_batch and len(kumpul) >= berikut_batch:
            simpan_batch(list(kumpul.values()))
            berikut_batch = (len(kumpul) // 100 + 1) * 100
            print(f"    [simpan] {len(kumpul)} ulasan sudah ditulis ke CSV",
                  flush=True)

        if len(kumpul) >= target:
            break
        scroll_panel_ulasan(driver)

    durasi = int(time.time() - detik_mulai)
    print(f"    -> {len(kumpul)} ulasan ({durasi}s)")
    return list(kumpul.values())

def baca_daftar_kafe(path):
    if not os.path.exists(path):
        sys.exit(f"File {path} tidak ada. Pastikan file tersedia.")
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def hapus_baris_kafe(path, place_id):
    """Hapus baris parsial milik satu kafe sebelum kafe diproses ulang.

    Baris ditulis bertahap per 100 ulasan, sehingga saat proses berhenti di
    tengah masih ada baris yang belum ditandai selesai. Baris lama dibuang
    lebih dulu agar tidak ganda ketika kafe diproses ulang.
    Mengembalikan jumlah baris yang dibuang.
    """
    if not place_id or not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        semua = list(csv.reader(f))
    if len(semua) < 2:
        return 0
    header, isi = semua[0], semua[1:]
    try:
        i_pid = header.index("placeId")
    except ValueError:
        return 0
    sisa = [r for r in isi if len(r) <= i_pid or r[i_pid] != place_id]
    buang = len(isi) - len(sisa)
    if buang:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(sisa)
    return buang

def teks_valid(teks: str, seen: set) -> bool:
    """Tiga kriteria pengecekan berurutan:
         1. teks kosong            -> False
         2. kurang dari 3 kata     -> False
         3. duplikat keep=first    -> False
    Pemeriksaan duplikat bersifat global pada seluruh dataset dan tanpa case
    folding, sehingga sama dengan perilaku pandas duplicated."""
    t = (teks or "").strip()
    if not t:                      # tahap 1: kosong
        return False
    if len(t.split()) < 3:         # tahap 2: <3 kata
        return False
    if t in seen:                  # tahap 3: duplikat
        return False
    seen.add(t)                    # kunci = teks mentah persis
    return True

def finalisasi_valid(path=None):
    """Hitung ulang kolom valid untuk seluruh file memakai pandas.

    Jalankan sekali setelah scraping selesai atau setelah dilanjutkan.
      python "scrapping Gmaps Nafii.py" --finalisasi
    """
    path = path or FILE_HASIL
    if not os.path.exists(path):
        sys.exit(f"{path} tidak ada.")

    import pandas as pd  # lokal, supaya tanpa pandas pun --help tetap jalan

    df = pd.read_csv(path, encoding="utf-8-sig")
    t = df["teks_ulasan"].fillna("").str.strip()

    m_kosong = t == ""
    sisa = df[~m_kosong]
    m_pendek = sisa["teks_ulasan"].str.strip().str.split().str.len().lt(3)
    sisa2 = sisa[~m_pendek]
    m_duplikat = sisa2.duplicated(subset=["teks_ulasan"], keep="first")
    valid_idx = sisa2[~m_duplikat].index

    df["valid"] = df.index.isin(valid_idx)
    df.to_csv(path, index=False, encoding="utf-8-sig")

    print("FINALISASI SELESAI")
    print(f"  Total awal        : {len(df)}")
    print(f"  Kosong            : {int(m_kosong.sum())}")
    print(f"  Kurang dari 3 kata: {int(m_pendek.sum())}")
    print(f"  Duplikat          : {int(m_duplikat.sum())}")
    print(f"  Data valid        : {int(df['valid'].sum())}")
    cek = m_kosong.sum() + m_pendek.sum() + m_duplikat.sum() + int(df["valid"].sum())
    print(f"  Cek penjumlahan   : {cek} (harus = {len(df)})")

def muat_progress():
    if os.path.exists(FILE_PROGRESS):
        try:
            return json.load(open(FILE_PROGRESS, encoding="utf-8"))
        except Exception:
            return []
    return []

def simpan_progress(done):
    json.dump(done, open(FILE_PROGRESS, "w", encoding="utf-8"), ensure_ascii=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2000,
                    help="batas atas ulasan per kafe (default 2000)")
    ap.add_argument("--mulai-dari", type=int, default=0,
                    help="lewati N kafe pertama (buat lanjut setelah berhenti)")
    ap.add_argument("--maks-kafe", type=int, default=0,
                    help="proses paling banyak N kafe lalu berhenti (0 = semua). "
                         "Tes cepat: --maks-kafe 1")
    ap.add_argument("--max", type=int, default=0,
                    help="batas ulasan PER kafe (mengganti --target)")
    ap.add_argument("--attach", action="store_true",
                    help="menempel ke Chrome yang sudah dibuka dan login "
                         "dengan port debug 9222")
    ap.add_argument("--port", type=int, default=9222,
                    help="port debugging untuk --attach (default 9222)")
    ap.add_argument("--finalisasi", action="store_true",
                    help="hitung ulang kolom valid untuk seluruh file (jalankan setelah beres)")
    args = ap.parse_args()

    if args.finalisasi:
        finalisasi_valid()
        return

    target = args.max or args.target  # --max dipakai saat tes

    daftar = baca_daftar_kafe(FILE_KAFE)
    done = set(muat_progress())
    print(f"Daftar kafe: {len(daftar)} | sudah selesai: {len(done)} | target/kafe: {target}")
    if args.attach:
        print(f"MODE: attach ke Chrome port {args.port} (harus sudah jalan + login)")
    if args.maks_kafe:
        print(f"Batas kafe run ini: {args.maks_kafe} (dari --mulai-dari {args.mulai_dari})")

    # set duplikat bertahan selama satu run karena duplikat dinilai lintas kafe
    teks_sudah_pakai = set()

    tulis_header = not os.path.exists(FILE_HASIL)
    fcsv = open(FILE_HASIL, "a", newline="", encoding="utf-8-sig")
    out = csv.DictWriter(fcsv, fieldnames=KOLOM)
    if tulis_header:
        out.writeheader()

    driver = bikin_driver(attach=args.attach, port=args.port)
    wait = WebDriverWait(driver, 20)
    berhenti_total = False
    diproses_run = 0

    try:
        for idx, kafe in enumerate(daftar):
            if idx < args.mulai_dari:
                continue

            link = kafe.get("link") or kafe.get("url") or ""
            if not link:
                continue
            pid = kafe.get("place_id") or place_id_dari_url(link)

            if pid and pid in done:
                print(f"[{idx+1}/{len(daftar)}] dilewati (sudah): {kafe.get('nama')}")
                continue

            if args.maks_kafe and diproses_run >= args.maks_kafe:
                print(f"Batas --maks-kafe {args.maks_kafe} tercapai; run ini berhenti.")
                break

            # dihitung sebelum dicoba, termasuk yang gagal, sehingga
            # --maks-kafe 1 berarti mencoba tepat satu kafe lalu berhenti
            diproses_run += 1
            print(f"[{idx+1}/{len(daftar)}] {kafe.get('nama')}")

            try:
                hasil_buka = buka_tab_ulasan(
                    driver, wait, link, nama_csv=kafe.get("nama", ""), place_id=pid)
            except WebDriverException as e:
                print(f"    ! error driver: {e}")
                break

            if hasil_buka is None:
                berhenti_total = True
                break
            if hasil_buka is False:
                # gagal membuka tab atau di luar Kota Malang, lanjut kafe berikutnya
                continue

            nama, alamat = hasil_buka
            if not alamat:
                # alamat gagal dibaca: tetap diproses, tapi ditandai
                print("    ! alamat tidak terbaca — tetap diproses, cek manual di CSV")

            if jumlah_ulasan(driver) == 0:
                print("    ! ulasan sama sekali tidak muncul, dilewati")
                continue

            kid = kafe_id_dari(nama, alamat or "")

            # buang baris parsial kafe ini bila pernah ditulis sebagian;
            # file CSV harus ditutup dulu sebelum dipangkas agar tidak
            # menimbulkan PermissionError di Windows
            fcsv.flush()
            fcsv.close()
            buang = hapus_baris_kafe(FILE_HASIL, pid)
            fcsv = open(FILE_HASIL, "a", newline="", encoding="utf-8-sig")
            out = csv.DictWriter(fcsv, fieldnames=KOLOM)
            if buang:
                print(f"    [hapus] {buang} baris parsial lama dibuang (diproses ulang)")

            # checkpoint per 100: tulis hanya row yang belum pernah ditulis,
            # dengan menghitung offset file untuk kafe ini
            def buat_simpan(pid=pid, nama=nama, kid=kid):
                posisi_terakhir = [0]  # jumlah row kafe ini yang sudah ada di file

                def simpan_batch(rows):
                    baru = rows[posisi_terakhir[0]:]
                    if not baru:
                        return
                    for r in baru:
                        out.writerow({
                            "placeId": pid,
                            "nama_kafe": nama,
                            "teks_ulasan": r["teks"],
                            "rating_ulasan": r["rating"],
                            "tanggal_ulasan": r["tanggal_abs"] or r["tanggal_rel"],
                            "kafe_id": kid,
                            "valid": str(teks_valid(r["teks"], teks_sudah_pakai)),
                        })
                    fcsv.flush()
                    posisi_terakhir[0] = len(rows)
                    print(f"    [simpan] {len(baru)} baris baru -> CSV "
                          f"(total {len(rows)} untuk kafe ini)", flush=True)

                return simpan_batch

            simpan = buat_simpan()
            rows = kumpul_ulasan(driver, nama, pid, kid,
                                 target=target, simpan_batch=simpan)
            # tulis sisa batch terakhir (yang belum kena checkpoint kelipatan 100)
            simpan(rows)
            fcsv.flush()

            # jangan tandai selesai bila 0 baris masuk, agar kafe tetap
            # dicoba lagi pada run berikutnya
            if not rows:
                print("    ! 0 ulasan terbaca TERSIMPAN — TIDAK ditandai selesai "
                      "(akan dicoba lagi di run berikutnya)")
                continue

            if pid:
                done.add(pid)
                simpan_progress(list(done))

            time.sleep(random.uniform(*JEDA_ANTAR_KAFE))

    finally:
        try:
            driver.quit()
        except Exception:
            pass
        fcsv.close()

    if berhenti_total:
        print("\nBerhenti karena CAPTCHA. Tunggu beberapa jam, lalu lanjutkan dengan: "
              'python "scrapping Gmaps Nafii.py" --mulai-dari N')
    else:
        print(f"\nSelesai. Hasil -> {FILE_HASIL}")


if __name__ == "__main__":
    main()
