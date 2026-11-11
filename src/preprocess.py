import json

from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE


def bersihkan(df):
    """Buang data kosong, buat label 'sukses' dan fitur 'tahun', ubah genre jadi 0/1."""
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)]
    df = df.dropna(subset=["runtime", "release_date"]).copy()

    # Label: sukses jika pendapatan lebih besar dari anggaran
    df["sukses"] = (df["revenue"] > df["budget"]).astype(int)
    df["tahun"] = df["release_date"].astype(str).str[:4].astype(int)

    # Genre: teks JSON -> kolom 0/1 per genre
    daftar = df["genres"].apply(lambda t: [g["name"] for g in json.loads(t)])
    genre = sorted({g for d in daftar for g in d})
    for g in genre:
        df["genre_" + g] = daftar.apply(lambda d: int(g in d))
    return df, genre


def siapkan_xy(df, genre):
    """Pilih fitur. 'revenue' TIDAK dipakai karena label dibuat darinya (data leakage)."""
    fitur = ["budget", "runtime", "tahun"] + ["genre_" + g for g in genre]
    X, y = df[fitur], df["sukses"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_tr, X_te, y_tr, y_te, fitur
