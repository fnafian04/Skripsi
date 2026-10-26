@echo off
title Nafinder App
cd /d "%~dp0"
echo Start Laragon dulu (Start All), lalu aplikasi akan dibuka.
set DB_HOST=localhost
set DB_PORT=3306
set DB_NAME=nafinder
set DB_USER=root
set DB_PASS=
python app.py
pause
