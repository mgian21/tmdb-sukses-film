# Prediksi Film Sukses atau Tidak Sukses (TMDB 5000)

Tugas besar Machine Learning. Klasifikasi film **sukses** (`revenue` > `budget`) atau **tidak sukses**
berdasarkan anggaran, durasi, tahun rilis, dan genre. Model: Logistic Regression, Decision Tree, Random Forest.

## Persiapan
1. Python 3.10 atau lebih baru.
2. Unduh **TMDB 5000 Movie Dataset** dari Kaggle. Ambil `tmdb_5000_movies.csv`.
3. Taruh file di `data/raw/tmdb_5000_movies.csv`.

## Instalasi
Windows:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Mac/Linux:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Jalankan
Pipeline (latih, evaluasi, simpan grafik dan tabel):
```
python main.py
```
Hasil ada di `outputs/figures/` dan `outputs/tables/hasil_perbandingan.csv`.

Dashboard interaktif:
```
streamlit run app.py
```

## Struktur
```
main.py            jalankan semua langkah
app.py             dashboard Streamlit
src/config.py      path dan pengaturan
src/data_loader.py baca CSV
src/preprocess.py  bersihkan data, label, fitur
src/train.py       latih 3 model
src/evaluate.py    metrik, grafik, tabel
```

## Catatan
- `revenue` tidak dipakai sebagai fitur (mencegah data leakage).
- Split 80/20 dengan `stratify` dan `random_state=42`.
- Ada baseline (tebak kelas mayoritas) sebagai pembanding.
