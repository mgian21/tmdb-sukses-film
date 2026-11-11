import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "data/raw/tmdb_5000_movies.csv"

st.set_page_config(page_title="Prediksi Film Sukses", page_icon="🎬", layout="wide")


@st.cache_data
def muat_data():
    df = pd.read_csv(DATA_PATH)
    # Buang data kosong
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)]
    df = df.dropna(subset=["runtime", "release_date"])
    # Label dan fitur tahun
    df["sukses"] = (df["revenue"] > df["budget"]).astype(int)
    df["tahun"] = pd.to_datetime(df["release_date"]).dt.year
    # Genre jadi kolom 0/1
    daftar = df["genres"].apply(lambda t: [g["name"] for g in json.loads(t)])
    genre = sorted({g for d in daftar for g in d})
    for g in genre:
        df["genre_" + g] = daftar.apply(lambda d: int(g in d))
    return df, genre


@st.cache_resource
def latih(_df, genre):
    fitur = ["budget", "runtime", "tahun"] + ["genre_" + g for g in genre]
    X, y = _df[fitur], _df["sukses"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    baris = []
    # Baseline: tebak kelas mayoritas
    mayoritas = int(y_tr.mode()[0])
    pred_base = [mayoritas] * len(y_te)
    baris.append(
        {
            "Model": "Baseline (tebak mayoritas)",
            "Accuracy": accuracy_score(y_te, pred_base),
            "Precision": precision_score(y_te, pred_base, zero_division=0),
            "Recall": recall_score(y_te, pred_base, zero_division=0),
            "F1": f1_score(y_te, pred_base, zero_division=0),
        }
    )
    for nama, m in models.items():
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        baris.append(
            {
                "Model": nama,
                "Accuracy": accuracy_score(y_te, p),
                "Precision": precision_score(y_te, p),
                "Recall": recall_score(y_te, p),
                "F1": f1_score(y_te, p),
            }
        )
    return models, X_te, y_te, fitur, pd.DataFrame(baris)


# ---------- Header ----------
st.title("🎬 Prediksi Film Sukses atau Tidak Sukses")
st.caption(
    "Dataset TMDB 5000 Movie · Logistic Regression, Decision Tree, Random Forest"
)

try:
    df, genre = muat_data()
except FileNotFoundError:
    st.error(f"File tidak ditemukan: {DATA_PATH}")
    st.stop()

models, X_te, y_te, fitur, tabel = latih(df, tuple(genre))
terbaik = tabel.iloc[1:].sort_values("F1", ascending=False).iloc[0]

# ---------- Kartu ringkasan ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah film", f"{len(df):,}")
c2.metric("Film sukses", f"{df['sukses'].mean() * 100:.1f}%")
c3.metric("Model terbaik (F1)", terbaik["Model"])
c4.metric("Accuracy terbaik", f"{terbaik['Accuracy'] * 100:.1f}%")

tab1, tab2, tab3 = st.tabs(["📊 Data", "🏆 Perbandingan Model", "🎯 Coba Prediksi"])

# ---------- Tab 1: Data ----------
with tab1:
    kiri, kanan = st.columns(2)
    with kiri:
        st.subheader("Sukses vs tidak sukses")
        hitung = df["sukses"].map({1: "Sukses", 0: "Tidak sukses"}).value_counts()
        st.bar_chart(hitung)
    with kanan:
        st.subheader("Jumlah film per genre")
        per_genre = df[["genre_" + g for g in genre]].sum()
        per_genre.index = genre
        st.bar_chart(per_genre.sort_values(ascending=False))
    st.subheader("Contoh data")
    st.dataframe(
        df[["title", "budget", "revenue", "runtime", "tahun", "sukses"]].head(10),
        use_container_width=True,
    )

# ---------- Tab 2: Perbandingan ----------
with tab2:
    st.subheader("Tabel perbandingan")
    st.dataframe(
        tabel.set_index("Model").style.format("{:.3f}").highlight_max(axis=0),
        use_container_width=True,
    )
    st.bar_chart(tabel.set_index("Model")[["Accuracy", "F1"]])

    kiri, kanan = st.columns(2)
    with kiri:
        st.subheader("Confusion matrix")
        pilih = st.selectbox("Pilih model", list(models.keys()))
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ConfusionMatrixDisplay.from_estimator(
            models[pilih],
            X_te,
            y_te,
            display_labels=["Tidak sukses", "Sukses"],
            cmap="Blues",
            ax=ax,
        )
        st.pyplot(fig)
    with kanan:
        st.subheader("Fitur paling berpengaruh (Random Forest)")
        penting = pd.Series(
            models["Random Forest"].feature_importances_, index=fitur
        ).sort_values(ascending=False)
        st.bar_chart(penting.head(8))

# ---------- Tab 3: Prediksi ----------
with tab3:
    st.subheader("Masukkan data film")
    kiri, kanan = st.columns(2)
    with kiri:
        budget_juta = st.slider("Anggaran (juta USD)", 1, 300, 50)
        runtime = st.slider("Durasi (menit)", 60, 240, 110)
        tahun = st.slider("Tahun rilis", 1980, 2025, 2015)
    with kanan:
        genre_pilih = st.multiselect("Genre", genre, default=genre[:1])

    if st.button("Prediksi", type="primary"):
        baris = {f: 0 for f in fitur}
        baris["budget"] = budget_juta * 1_000_000
        baris["runtime"] = runtime
        baris["tahun"] = tahun
        for g in genre_pilih:
            baris["genre_" + g] = 1
        x = pd.DataFrame([baris])[fitur]
        peluang = models["Random Forest"].predict_proba(x)[0][1]
        st.progress(float(peluang))
        if peluang >= 0.5:
            st.success(f"🎉 Kemungkinan SUKSES: {peluang * 100:.1f}%")
        else:
            st.warning(f"⚠️ Kemungkinan TIDAK SUKSES: {(1 - peluang) * 100:.1f}%")
        st.caption("Prediksi dari Random Forest. Hanya untuk keperluan tugas kuliah.")
