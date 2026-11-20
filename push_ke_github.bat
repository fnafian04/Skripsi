@echo off
cd /d "%~dp0"
echo ===================================================
echo   Mengunggah Proyek Nafinder ke GitHub (Skripsi)
echo ===================================================
echo.

git remote get-url origin >nul 2>&1
if %errorlevel% neq 0 (
    echo Remote 'origin' belum terpasang.
    echo Buat repository baru di GitHub, lalu masukkan URL-nya di bawah ini.
    echo Contoh: https://github.com/username/nafinder.git
    echo.
    set /p REPO_URL="URL GitHub Repository: "
    if defined REPO_URL (
        git remote add origin %REPO_URL%
    ) else (
        echo [!] URL tidak boleh kosong. Dibatalkan.
        pause
        exit /b 1
    )
)

echo Menambahkan perubahan terbaru...
git add .
git commit -m "update: prepare web deployment on Vercel with Supabase PostgreSQL" >nul 2>&1

echo Mengunggah ke branch main...
git push -u origin main

echo.
echo ===================================================
echo Selesai! Silakan cek repositori GitHub kamu.
echo Selanjutnya sambungkan ke Vercel untuk deploy otomatis.
echo ===================================================
pause
