import os
from flask import Flask, redirect, url_for
from db import Base
import models  # register ORM models
from routes import auth, dashboard, dataset, hasil, preprocessing, rekomendasi, dokumentasi


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-ganti-jadi-acak')

    aspek_labels = {
        'makanan_minuman': 'Makanan dan Minuman',
        'pelayanan': 'Pelayanan',
        'harga': 'Harga',
        'suasana': 'Suasana',
        'fasilitas_wfc': 'Fasilitas WFC',
        'kebersihan': 'Kebersihan',
    }

    @app.template_filter('format_aspek')
    def format_aspek(value):
        key = str(value or '').strip()
        return aspek_labels.get(key, key.replace('_', ' ').title())

    app.register_blueprint(auth.bp)
    app.register_blueprint(dashboard.bp)
    app.register_blueprint(dataset.bp)
    app.register_blueprint(preprocessing.bp)
    app.register_blueprint(hasil.bp)
    app.register_blueprint(rekomendasi.bp)
    app.register_blueprint(dokumentasi.bp)

    @app.route('/')
    def root():
        return redirect(url_for('dashboard.bantuan'))
    return app

if __name__ == '__main__':
    app=create_app()
    app.run(debug=True,port=5000)
