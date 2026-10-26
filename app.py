import os
from flask import Flask, redirect, url_for
from db import Base
import models  # register ORM models
from routes import auth, dashboard, dataset, hasil, rekomendasi, dokumentasi, validasi, validasi_v4, validasi_v5, riwayat_labeling


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-ganti-jadi-acak')
    app.register_blueprint(auth.bp)
    app.register_blueprint(dashboard.bp)
    app.register_blueprint(dataset.bp)
    app.register_blueprint(hasil.bp)
    app.register_blueprint(rekomendasi.bp)
    app.register_blueprint(dokumentasi.bp)
    app.register_blueprint(validasi.bp)
    app.register_blueprint(validasi_v4.bp)
    app.register_blueprint(validasi_v5.bp)
    app.register_blueprint(riwayat_labeling.bp)

    @app.route('/')
    def root():
        return redirect(url_for('dashboard.index'))
    return app

if __name__ == '__main__':
    app=create_app()
    app.run(debug=True,port=5000)
