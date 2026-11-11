import pandas as pd

from src.config import DATA_PATH


def load_data():
    """Baca CSV TMDB. Beri pesan jelas jika file belum ada."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {DATA_PATH}\n"
            "Unduh 'TMDB 5000 Movie Dataset' dari Kaggle, lalu taruh "
            "tmdb_5000_movies.csv di folder data/raw/"
        )
    return pd.read_csv(DATA_PATH)
