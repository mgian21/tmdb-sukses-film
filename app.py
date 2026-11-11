import json

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "data/raw/tmdb_5000_movies.csv"

st.set_page_config(
    page_title="Proyeksi Box Office TMDB 5000",
    layout="wide",
)

# Gaya visual: Ruang pemutaran sinema & jurnal analisis Box Office (DESIGN.md)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,700;1,400&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    :root {
        --bg-obsidian: #0D0F12;
        --bg-charcoal: #161920;
        --border-slate: #262B36;
        --text-ivory: #F4F1EA;
        --text-silver: #B8B2A6;
        --accent-gold: #E5A93C;
        --status-hit: #4CAF7D;
        --status-flop: #D9534F;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-ivory);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    h1, h2, h3, .playbill-title {
        font-family: 'Playfair Display', Georgia, serif !important;
        color: var(--text-ivory) !important;
        letter-spacing: -0.01em;
    }

    /* Motif identitas: Tajuk Playbill Sinema dengan garis ganda */
    .playbill-header {
        border-top: 2px solid var(--accent-gold);
        border-bottom: 1px solid var(--border-slate);
        padding: 1.25rem 0 1.1rem 0;
        margin-bottom: 1.75rem;
    }

    .playbill-title {
        font-size: clamp(1.85rem, 3.2vw, 2.65rem);
        font-weight: 700;
        line-height: 1.15;
        margin: 0 0 0.45rem 0;
    }

    .playbill-subtitle {
        font-size: 0.98rem;
        color: var(--text-silver);
        margin: 0;
        line-height: 1.5;
    }

    /* Baris konteks arsip dengan angka tabular (bukan kartu template seragam) */
    .archive-ledger {
        display: flex;
        flex-wrap: wrap;
        gap: 1.25rem 2.25rem;
        background-color: var(--bg-charcoal);
        border: 1px solid var(--border-slate);
        border-radius: 6px;
        padding: 1rem 1.35rem;
        margin-bottom: 2rem;
        font-variant-numeric: tabular-nums;
    }

    .ledger-item {
        flex: 1 1 160px;
        min-width: 140px;
    }

    .ledger-label {
        font-size: 0.8rem;
        color: var(--text-silver);
        margin-bottom: 0.2rem;
        font-weight: 500;
    }

    .ledger-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-ivory);
    }

    .ledger-sub {
        font-size: 0.78rem;
        color: var(--text-silver);
        margin-top: 0.15rem;
    }

    .ledger-value.gold {
        color: var(--accent-gold);
    }

    /* Motif identitas: Panel Potongan Tiket (Ticket Stub) */
    .ticket-panel {
        background-color: var(--bg-charcoal);
        border: 1px solid var(--border-slate);
        border-top: 3px solid var(--accent-gold);
        border-radius: 6px;
        padding: 1.4rem 1.5rem;
        font-variant-numeric: tabular-nums;
    }

    .ticket-panel.hit {
        border-top-color: var(--status-hit);
    }

    .ticket-panel.flop {
        border-top-color: var(--status-flop);
    }

    .ticket-perforation {
        border-top: 1px dashed #383F4E;
        margin: 1rem 0;
        padding-top: 1rem;
    }

    .verdict-heading {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.45rem;
        font-weight: 700;
        margin: 0.2rem 0 0.5rem 0;
    }

    .verdict-hit {
        color: var(--status-hit);
    }

    .verdict-flop {
        color: var(--status-flop);
    }

    .ticket-meta {
        font-size: 0.9rem;
        color: var(--text-silver);
        line-height: 1.55;
        margin-bottom: 0.5rem;
    }

    .model-row {
        display: flex;
        justify-content: space-between;
        padding: 0.35rem 0;
        border-bottom: 1px solid #1F242E;
        font-size: 0.88rem;
    }

    .model-row:last-child {
        border-bottom: none;
    }

    /* Tombol utama & aksesibilitas fokus keyboard (R-03, R-11, R-25, R-32) */
    div.stButton > button[kind="primary"] {
        background-color: var(--accent-gold) !important;
        color: #0D0F12 !important;
        border: 1px solid var(--accent-gold) !important;
        border-radius: 4px !important;
        font-weight: 700 !important;
        min-height: 44px !important;
        padding: 0.55rem 1.25rem !important;
        transition: background-color 0.15s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #F0B84D !important;
        color: #0D0F12 !important;
    }

    button:focus-visible,
    input:focus-visible,
    [role="tab"]:focus-visible,
    [data-baseweb="select"] *:focus-visible {
        outline: 2px solid var(--accent-gold) !important;
        outline-offset: 2px !important;
    }

    button[role="tab"] {
        min-height: 44px !important;
        font-weight: 600 !important;
    }

    .section-intro {
        color: var(--text-silver);
        font-size: 0.92rem;
        margin-top: -0.35rem;
        margin-bottom: 1.1rem;
    }

    .state-box {
        background-color: var(--bg-charcoal);
        border: 1px solid var(--border-slate);
        border-radius: 6px;
        padding: 0.9rem 1.15rem;
        color: var(--text-silver);
        font-size: 0.89rem;
        line-height: 1.5;
        margin-bottom: 0.9rem;
    }

    .state-box.pending {
        border-color: var(--accent-gold);
        color: var(--text-ivory);
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .archive-ledger {
            gap: 1rem;
            padding: 0.85rem 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _urai_genre_teks(teks_json):
    """Ubah string JSON kolom genres menjadi daftar nama genre dengan aman."""
    if pd.isna(teks_json):
        return []
    try:
        return [g["name"] for g in json.loads(teks_json) if "name" in g]
    except Exception:
        return []


@st.cache_data
def muat_data():
    df_mentah = pd.read_csv(DATA_PATH)

    # Siapkan kolom tambahan pada data mentah (4.803 film) untuk eksplorasi arsip penuh
    df_mentah["daftar_genre"] = df_mentah["genres"].apply(_urai_genre_teks)
    df_mentah["genre_teks"] = df_mentah["daftar_genre"].apply(
        lambda d: ", ".join(d) if d else "Tidak Tercatat"
    )
    df_mentah["tahun_rilis"] = pd.to_datetime(
        df_mentah["release_date"], errors="coerce"
    ).dt.year
    df_mentah["selisih_usd"] = df_mentah["revenue"] - df_mentah["budget"]

    def _status_mentah(baris):
        if (
            baris["budget"] <= 0
            or baris["revenue"] <= 0
            or pd.isna(baris["runtime"])
            or pd.isna(baris["tahun_rilis"])
        ):
            return "Data Tidak Lengkap ($0)"
        return (
            "Untung (Hit)"
            if baris["revenue"] > baris["budget"]
            else "Rugi (Flop)"
        )

    df_mentah["status_arsip"] = df_mentah.apply(_status_mentah, axis=1)

    # Filter data bersih (budget > 0, revenue > 0, runtime & release_date valid) untuk model ML
    df = df_mentah[
        (df_mentah["budget"] > 0) & (df_mentah["revenue"] > 0)
    ].dropna(subset=["runtime", "release_date"]).copy()

    df["sukses"] = (df["revenue"] > df["budget"]).astype(int)
    df["tahun"] = df["tahun_rilis"].astype(int)

    genre = sorted({g for d in df["daftar_genre"] for g in d})
    for g in genre:
        df["genre_" + g] = df["daftar_genre"].apply(lambda d: int(g in d))

    return df, df_mentah, genre


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

    total_rugi_uji = int((y_te == 0).sum())
    baris = []

    # Baseline: selalu tebak kelas mayoritas pada data latih
    mayoritas = int(y_tr.mode()[0])
    pred_base = np.array([mayoritas] * len(y_te))
    tn_base = int(((y_te == 0) & (pred_base == 0)).sum())
    baris.append(
        {
            "Model": "Baseline (tebak mayoritas)",
            "Accuracy": accuracy_score(y_te, pred_base),
            "Precision": precision_score(y_te, pred_base, zero_division=0),
            "Recall": recall_score(y_te, pred_base, zero_division=0),
            "F1": f1_score(y_te, pred_base, zero_division=0),
            "Deteksi Film Rugi": f"{tn_base} / {total_rugi_uji} ({tn_base / total_rugi_uji * 100:.1f}%)",
        }
    )
    for nama, m in models.items():
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        tn = int(((y_te == 0) & (p == 0)).sum())
        baris.append(
            {
                "Model": nama,
                "Accuracy": accuracy_score(y_te, p),
                "Precision": precision_score(y_te, p),
                "Recall": recall_score(y_te, p),
                "F1": f1_score(y_te, p),
                "Deteksi Film Rugi": f"{tn} / {total_rugi_uji} ({tn / total_rugi_uji * 100:.1f}%)",
            }
        )
    return models, X_te, y_te, fitur, pd.DataFrame(baris)


def gambar_confusion_matrix(model, X_te, y_te, nama_model):
    """Buat plot Confusion Matrix dengan kontras tinggi sesuai tema ruang bioskop (R-25)."""
    y_pred = model.predict(X_te)
    cm = confusion_matrix(y_te, y_pred)

    cmap_cinema = mcolors.LinearSegmentedColormap.from_list(
        "cinema_gold", ["#1B1F29", "#7C5B24", "#E5A93C"]
    )
    fig, ax = plt.subplots(figsize=(4.4, 3.5), facecolor="#161920")
    ax.set_facecolor("#161920")
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap_cinema)

    labels = ["Rugi (Flop)", "Untung (Hit)"]
    ax.set_xticks([0, 1], labels=labels, color="#F4F1EA", fontsize=9.5)
    ax.set_yticks([0, 1], labels=labels, color="#F4F1EA", fontsize=9.5)
    ax.set_xlabel("Prediksi Model", color="#B8B2A6", fontsize=9.5, labelpad=8)
    ax.set_ylabel("Kenyataan Box Office", color="#B8B2A6", fontsize=9.5, labelpad=8)
    ax.set_title(
        f"Matriks Evaluasi: {nama_model}",
        color="#F4F1EA",
        fontsize=10.5,
        fontweight="bold",
        pad=12,
    )

    for spine in ax.spines.values():
        spine.set_edgecolor("#262B36")

    batas = (cm.max() + cm.min()) / 2.0
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            nilai = cm[i, j]
            persen = (nilai / total) * 100 if total > 0 else 0.0
            warna_teks = "#0D0F12" if nilai > batas else "#F4F1EA"
            ax.text(
                j,
                i,
                f"{nilai}\n({persen:.1f}%)",
                ha="center",
                va="center",
                color=warna_teks,
                fontsize=10,
                fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#B8B2A6", labelcolor="#B8B2A6", labelsize=8)
    cbar.outline.set_edgecolor("#262B36")
    fig.tight_layout()
    return fig


def gambar_batang_horizontal(series_data, warna_batang, format_nilai="{:.0f}", tinggi=3.2):
    """Buat grafik batang horizontal terurut dari nilai terbesar di atas (mengatasi urutan abjad st.bar_chart)."""
    s = series_data.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(5.2, tinggi), facecolor="#161920")
    ax.set_facecolor("#161920")

    bars = ax.barh(s.index, s.values, color=warna_batang, height=0.62)
    ax.tick_params(axis="y", colors="#F4F1EA", labelsize=9)
    ax.tick_params(axis="x", colors="#B8B2A6", labelsize=8.5)
    ax.xaxis.grid(True, linestyle="--", alpha=0.25, color="#B8B2A6")
    ax.set_axisbelow(True)

    maks = float(s.max()) if len(s) > 0 else 1.0
    ax.set_xlim(0, maks * 1.18)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_edgecolor("#262B36")

    for bar in bars:
        lebar = bar.get_width()
        ax.text(
            lebar + maks * 0.02,
            bar.get_y() + bar.get_height() / 2,
            format_nilai.format(lebar),
            va="center",
            ha="left",
            color="#F4F1EA",
            fontsize=8.5,
            fontweight="bold",
        )

    fig.tight_layout()
    return fig


def gambar_perbandingan_metrik(df_tabel):
    """Buat grafik perbandingan Accuracy dan F1 yang mempertahankan urutan baris tabel."""
    fig, ax = plt.subplots(figsize=(5.6, 3.0), facecolor="#161920")
    ax.set_facecolor("#161920")

    nama_model = df_tabel["Model"].tolist()[::-1]
    acc = df_tabel["Accuracy"].tolist()[::-1]
    f1 = df_tabel["F1"].tolist()[::-1]

    y_pos = np.arange(len(nama_model))
    tinggi = 0.34

    b1 = ax.barh(y_pos + tinggi / 2, acc, height=tinggi, label="Accuracy", color="#B8B2A6")
    b2 = ax.barh(y_pos - tinggi / 2, f1, height=tinggi, label="Skor F1", color="#E5A93C")

    ax.set_yticks(y_pos, labels=nama_model, color="#F4F1EA", fontsize=9)
    ax.tick_params(axis="x", colors="#B8B2A6", labelsize=8.5)
    ax.set_xlim(0, 1.05)
    ax.xaxis.grid(True, linestyle="--", alpha=0.25, color="#B8B2A6")
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_edgecolor("#262B36")

    for bars in [b1, b2]:
        for bar in bars:
            w = bar.get_width()
            ax.text(
                w + 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}",
                va="center",
                ha="left",
                color="#F4F1EA",
                fontsize=8,
            )

    leg = ax.legend(
        loc="lower right",
        facecolor="#0D0F12",
        edgecolor="#262B36",
        labelcolor="#F4F1EA",
        fontsize=8.5,
    )
    fig.tight_layout()
    return fig


# ---------- Tajuk Playbill Sinema ----------
st.markdown(
    """
    <div class="playbill-header">
        <h1 class="playbill-title">Proyeksi Box Office & Analisis Kesuksesan Film</h1>
        <p class="playbill-subtitle">
            Evaluasi peluang balik modal produksi film (pendapatan melampaui anggaran) berdasarkan arsip
            TMDB 5000 menggunakan Logistic Regression, Decision Tree, dan Random Forest.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Loading & Error States (R-27) ----------
try:
    with st.spinner(
        "Memuat arsip film TMDB 5000 dan melatih tiga model klasifikasi Box Office..."
    ):
        df, df_mentah, genre = muat_data()
        if df.empty:
            st.error(
                "Arsip data kosong setelah pembersihan (tidak ada baris dengan anggaran dan pendapatan valid). "
                "Periksa kembali isi file data/raw/tmdb_5000_movies.csv lalu muat ulang halaman."
            )
            st.stop()
        models, X_te, y_te, fitur, tabel = latih(df, tuple(genre))
except FileNotFoundError:
    st.error(
        f"File dataset tidak ditemukan di `{DATA_PATH}`. "
        "Unduh `tmdb_5000_movies.csv` dari Kaggle, simpan di dalam folder `data/raw/`, "
        "lalu muat ulang halaman ini."
    )
    st.stop()
except Exception as exc:
    st.error(
        f"Gagal membaca atau memproses arsip film: {exc}. "
        "Pastikan format kolom `budget`, `revenue`, `runtime`, `release_date`, dan `genres` sesuai standar TMDB 5000."
    )
    st.stop()

terbaik_acc = tabel.iloc[1:].sort_values("Accuracy", ascending=False).iloc[0]
tahun_min_data = int(df["tahun"].min())
tahun_maks_data = int(df["tahun"].max())

# ---------- Ringkasan Arsip (Ledger Bar) ----------
st.markdown(
    f"""
    <div class="archive-ledger" role="region" aria-label="Ringkasan Arsip TMDB">
        <div class="ledger-item">
            <div class="ledger-label">Arsip Bersih Terverifikasi</div>
            <div class="ledger-value">{len(df):,} judul</div>
            <div class="ledger-sub">Dari {len(df_mentah):,} total baris CSV ({tahun_min_data}-{tahun_maks_data})</div>
        </div>
        <div class="ledger-item">
            <div class="ledger-label">Proporsi Balik Modal (Hit)</div>
            <div class="ledger-value">{df['sukses'].mean() * 100:.1f}%</div>
            <div class="ledger-sub">{int(df['sukses'].sum()):,} Hit vs {int((df['sukses'] == 0).sum()):,} Flop</div>
        </div>
        <div class="ledger-item">
            <div class="ledger-label">Akurasi Model Tertinggi</div>
            <div class="ledger-value gold">{terbaik_acc['Model']}</div>
            <div class="ledger-sub">Accuracy {terbaik_acc['Accuracy'] * 100:.1f}% | F1 {terbaik_acc['F1']:.3f}</div>
        </div>
        <div class="ledger-item">
            <div class="ledger-label">Skema Evaluasi Uji</div>
            <div class="ledger-value">Split 80 / 20</div>
            <div class="ledger-sub">{len(X_te):,} film uji terstratifikasi</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Bagian 1 (Focal Point): Studio Proyeksi Box Office ----------
st.subheader("Studio Proyeksi Box Office")
st.markdown(
    '<p class="section-intro">'
    "Atur rencana produksi film Anda di sebelah kiri, lalu tekan tombol proyeksi untuk mencetak tiket evaluasi di sebelah kanan."
    "</p>",
    unsafe_allow_html=True,
)

kol_input, kol_tiket = st.columns([1.15, 0.85], gap="large")

with kol_input:
    c_in1, c_in2 = st.columns(2)
    with c_in1:
        budget_juta = st.slider(
            "Anggaran Produksi (Juta USD)", min_value=1, max_value=300, value=50
        )
        runtime = st.slider(
            "Durasi Tayang (Menit)", min_value=60, max_value=240, value=115
        )
    with c_in2:
        tahun = st.slider(
            "Tahun Rilis Target",
            min_value=tahun_min_data,
            max_value=2026,
            value=2016,
            help=f"Data historis di arsip TMDB 5000 mencakup film rilis tahun {tahun_min_data} hingga {tahun_maks_data}.",
        )
        model_utama = st.selectbox(
            "Model Penentu Utama",
            options=list(models.keys()),
            index=2,
            help="Pilih algoritma yang menjadi penentu utama pada tiket evaluasi. Konsensus ketiga model tetap ditampilkan di bawah tiket.",
        )

    genre_pilih = st.multiselect(
        "Genre Film",
        options=genre,
        default=["Action", "Adventure"]
        if "Action" in genre and "Adventure" in genre
        else genre[:1],
        placeholder="Pilih satu atau beberapa genre...",
    )

    if len(genre_pilih) == 0:
        st.markdown(
            '<div class="state-box">'
            "<strong>Belum ada genre yang dipilih.</strong> Pilih minimal satu genre di atas agar model "
            "memperhitungkan karakteristik pasar genre tersebut, atau tekan tombol di bawah untuk menguji "
            "hanya dari anggaran, durasi, dan tahun rilis."
            "</div>",
            unsafe_allow_html=True,
        )

    hitung_klik = st.button("Hitung Proyeksi Box Office", type="primary")
    if hitung_klik or "proyeksi_terakhir" not in st.session_state:
        baris_uji = {f: 0 for f in fitur}
        baris_uji["budget"] = budget_juta * 1_000_000
        baris_uji["runtime"] = runtime
        baris_uji["tahun"] = tahun
        for g in genre_pilih:
            baris_uji["genre_" + g] = 1
        x_uji = pd.DataFrame([baris_uji])[fitur]

        peluang_model = {}
        for nama_m, m_obj in models.items():
            peluang_model[nama_m] = float(m_obj.predict_proba(x_uji)[0][1])

        if genre_pilih:
            mask_genre = df[["genre_" + g for g in genre_pilih]].any(axis=1)
            df_sub = df.loc[mask_genre]
        else:
            df_sub = df

        median_budget_genre = float(df_sub["budget"].median() / 1_000_000)
        rata_sukses_genre = float(df_sub["sukses"].mean() * 100)
        jumlah_film_genre = int(len(df_sub))

        st.session_state["proyeksi_terakhir"] = {
            "budget_juta": budget_juta,
            "runtime": runtime,
            "tahun": tahun,
            "model_utama": model_utama,
            "genre": list(genre_pilih),
            "peluang_utama": peluang_model[model_utama],
            "semua_peluang": peluang_model,
            "median_budget_genre": median_budget_genre,
            "rata_sukses_genre": rata_sukses_genre,
            "jumlah_film_genre": jumlah_film_genre,
        }

with kol_tiket:
    hasil = st.session_state["proyeksi_terakhir"]

    # Deteksi apakah pengguna telah menggeser input namun belum menekan tombol hitung ulang
    parameter_berubah = (
        hasil["budget_juta"] != budget_juta
        or hasil["runtime"] != runtime
        or hasil["tahun"] != tahun
        or hasil["model_utama"] != model_utama
        or hasil["genre"] != list(genre_pilih)
    )
    if parameter_berubah:
        st.markdown(
            '<div class="state-box pending">'
            "<strong>Parameter di sebelah kiri telah diubah.</strong> "
            "Tekan tombol <strong>Hitung Proyeksi Box Office</strong> untuk memperbarui tiket evaluasi di bawah."
            "</div>",
            unsafe_allow_html=True,
        )

    peluang_utama = hasil["peluang_utama"]
    is_hit = peluang_utama >= 0.5
    kelas_tiket = "hit" if is_hit else "flop"
    kelas_judul = "verdict-hit" if is_hit else "verdict-flop"
    status_teks = (
        "PROYEKSI: BOX OFFICE HIT (UNTUNG)"
        if is_hit
        else "PROYEKSI: BOX OFFICE FLOP (RUGI)"
    )
    label_genre = (
        ", ".join(hasil["genre"]) if hasil["genre"] else "Tanpa Genre Spesifik"
    )

    baris_html_model = "".join(
        [
            f'<div class="model-row">'
            f'<span>{nama_m}{" (Utama)" if nama_m == hasil["model_utama"] else ""}</span>'
            f'<strong>{prob * 100:.1f}% peluang untung</strong>'
            f"</div>"
            for nama_m, prob in hasil["semua_peluang"].items()
        ]
    )

    st.markdown(
        f"""
        <div class="ticket-panel {kelas_tiket}" role="status" aria-live="polite">
            <div class="ledger-label">Tiket Evaluasi Produksi (Target Rilis {hasil['tahun']})</div>
            <div class="verdict-heading {kelas_judul}">{status_teks}</div>
            <div class="ticket-meta">
                Peluang balik modal menurut <strong>{hasil['model_utama']}</strong>:
                <strong>{peluang_utama * 100:.1f}%</strong><br>
                Spesifikasi: Anggaran <strong>USD {hasil['budget_juta']} Juta</strong>,
                Durasi <strong>{hasil['runtime']} menit</strong>,
                Genre: <strong>{label_genre}</strong>.
            </div>
            <div class="ticket-perforation">
                <div class="ledger-label">Pembanding Arsip TMDB ({hasil['jumlah_film_genre']:,} film sejenis)</div>
                <div class="ticket-meta">
                    Median anggaran historis: <strong>USD {hasil['median_budget_genre']:.1f} Juta</strong>
                    (Tingkat balik modal historis: <strong>{hasil['rata_sukses_genre']:.1f}%</strong>).
                </div>
            </div>
            <div class="ticket-perforation">
                <div class="ledger-label">Konsensus Tiga Model Klasifikasi</div>
                {baris_html_model}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(float(peluang_utama))

st.divider()

# ---------- Bagian 2: Bukti Evaluasi & Eksplorasi Arsip ----------
tab_eval, tab_data = st.tabs(
    ["Evaluasi & Perbandingan Model", "Eksplorasi Arsip TMDB 5000"]
)

with tab_eval:
    eval_kiri, eval_kanan = st.columns([1.2, 0.8], gap="large")

    with eval_kiri:
        st.subheader("Tabel Perbandingan Metrik")
        st.markdown(
            '<p class="section-intro">'
            "Evaluasi pada 20% data uji (646 film: 488 Hit dan 158 Flop). Perhatikan kolom 'Deteksi Film Rugi' untuk melihat kemampuan model menyaring potensi kerugian."
            "</p>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            tabel.set_index("Model").style.format(
                {
                    "Accuracy": "{:.3f}",
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1": "{:.3f}",
                }
            ),
            width="stretch",
        )
        st.markdown(
            '<div class="state-box">'
            "<strong>Catatan Analisis Evaluasi:</strong> "
            "Karena 75,5% film di arsip bersih merupakan film untung, <em>Baseline (tebak mayoritas)</em> yang menebak seluruh film selalu untung "
            "memperoleh Recall 1,000 tetapi gagal mendeteksi satu pun film yang merugi (0 dari 158 film rugi). "
            "<strong>Decision Tree</strong> dan <strong>Random Forest</strong> memiliki Precision lebih tinggi dan mampu mendeteksi puluhan film yang berisiko merugi."
            "</div>",
            unsafe_allow_html=True,
        )

        st.subheader("Seberapa jauh model melampaui tebakan mayoritas (Baseline)?")
        fig_metrik = gambar_perbandingan_metrik(tabel)
        st.pyplot(fig_metrik, width="stretch")
        plt.close(fig_metrik)

    with eval_kanan:
        st.subheader("Diagnosis Kesalahan Prediksi (Confusion Matrix)")
        pilih_model = st.selectbox(
            "Pilih model untuk melihat rincian tebakan benar vs meleset:",
            list(models.keys()),
            index=2,
        )
        fig_cm = gambar_confusion_matrix(models[pilih_model], X_te, y_te, pilih_model)
        st.pyplot(fig_cm, width="stretch")
        plt.close(fig_cm)

        st.subheader("Fitur apa yang paling menentukan proyeksi balik modal?")
        st.caption(
            "Delapan fitur dengan bobot kepentingan (feature importance) tertinggi pada model Random Forest:"
        )
        penting = (
            pd.Series(models["Random Forest"].feature_importances_, index=fitur)
            .sort_values(ascending=False)
            .head(8)
        )
        penting.index = [
            idx.replace("genre_", "Genre: ")
            .replace("budget", "Anggaran (Budget)")
            .replace("runtime", "Durasi (Runtime)")
            .replace("tahun", "Tahun Rilis")
            for idx in penting.index
        ]
        fig_imp = gambar_batang_horizontal(
            penting, warna_batang="#E5A93C", format_nilai="{:.3f}", tinggi=3.3
        )
        st.pyplot(fig_imp, width="stretch")
        plt.close(fig_imp)

with tab_data:
    arsip_kiri, arsip_kanan = st.columns([0.85, 1.15], gap="large")

    with arsip_kiri:
        st.subheader("Berapa banyak film di arsip TMDB yang berhasil balik modal?")
        st.caption(
            f"Distribusi pada {len(df):,} film yang memiliki catatan anggaran dan pendapatan lengkap:"
        )
        hitung_status = (
            df["sukses"]
            .map({1: "Untung (Hit)", 0: "Rugi (Flop)"})
            .value_counts()
        )
        fig_status = gambar_batang_horizontal(
            hitung_status, warna_batang="#E5A93C", format_nilai="{:,.0f} film", tinggi=2.1
        )
        st.pyplot(fig_status, width="stretch")
        plt.close(fig_status)

        st.subheader("Genre apa yang paling banyak diproduksi dalam arsip TMDB?")
        st.caption("12 genre dengan jumlah produksi film terbanyak pada arsip bersih:")
        per_genre = df[["genre_" + g for g in genre]].sum()
        per_genre.index = genre
        fig_genre = gambar_batang_horizontal(
            per_genre.sort_values(ascending=False).head(12),
            warna_batang="#B8B2A6",
            format_nilai="{:,.0f}",
            tinggi=3.8,
        )
        st.pyplot(fig_genre, width="stretch")
        plt.close(fig_genre)

    with arsip_kanan:
        st.subheader("Telusuri Data Film di Arsip TMDB")

        mode_arsip = st.radio(
            "Cakupan Data Arsip",
            options=[
                f"Arsip Bersih Model ({len(df):,} film dengan anggaran & pendapatan > $0)",
                f"Seluruh Data Mentah CSV ({len(df_mentah):,} film, termasuk data $0 / belum tercatat)",
            ],
            index=0,
            horizontal=False,
        )

        pakai_mentah = mode_arsip.startswith("Seluruh Data Mentah")
        df_sumber = df_mentah if pakai_mentah else df

        f_col1, f_col2, f_col3 = st.columns([1.2, 1.0, 1.0])
        with f_col1:
            cari_judul = st.text_input(
                "Cari judul film",
                value="",
                placeholder="Misal: Avatar, Spider-Man...",
            )
        with f_col2:
            opsi_status = (
                [
                    "Semua Film",
                    "Untung (Hit)",
                    "Rugi (Flop)",
                    "Data Tidak Lengkap ($0)",
                ]
                if pakai_mentah
                else ["Semua Film", "Untung (Hit)", "Rugi (Flop)"]
            )
            filter_status = st.selectbox("Filter status Box Office", opsi_status)
        with f_col3:
            filter_genre_arsip = st.selectbox(
                "Filter genre",
                ["Semua Genre"] + genre,
            )

        df_tampil = df_sumber.copy()
        if cari_judul.strip():
            df_tampil = df_tampil[
                df_tampil["title"].str.contains(
                    cari_judul.strip(), case=False, na=False
                )
            ]
        if filter_status != "Semua Film":
            df_tampil = df_tampil[df_tampil["status_arsip"] == filter_status]
        if filter_genre_arsip != "Semua Genre":
            df_tampil = df_tampil[
                df_tampil["daftar_genre"].apply(lambda d: filter_genre_arsip in d)
            ]

        if df_tampil.empty:
            st.markdown(
                f'<div class="state-box">'
                f"<strong>Tidak ada film yang cocok dengan pencarian Anda.</strong> "
                f"Penyebab: kombinasi kata kunci judul, status, atau genre tidak ditemukan di dalam {len(df_sumber):,} film pada cakupan ini. "
                f"Hapus kata kunci di kolom pencarian atau kembalikan filter ke 'Semua' untuk menampilkan data."
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption(
                f"Menampilkan seluruh {len(df_tampil):,} film yang sesuai dari cakupan yang dipilih ({len(df_sumber):,} total film):"
            )
            tabel_film = df_tampil[
                [
                    "title",
                    "tahun_rilis",
                    "genre_teks",
                    "budget",
                    "revenue",
                    "selisih_usd",
                    "runtime",
                    "status_arsip",
                ]
            ].copy()
            tabel_film["tahun_rilis"] = (
                tabel_film["tahun_rilis"]
                .fillna(0)
                .astype(int)
                .replace(0, np.nan)
            )
            tabel_film = tabel_film.rename(
                columns={
                    "title": "Judul Film",
                    "tahun_rilis": "Tahun",
                    "genre_teks": "Genre",
                    "budget": "Anggaran (USD)",
                    "revenue": "Pendapatan (USD)",
                    "selisih_usd": "Selisih Laba/Rugi (USD)",
                    "runtime": "Durasi (Menit)",
                    "status_arsip": "Status Box Office",
                }
            )
            st.dataframe(tabel_film, width="stretch", hide_index=True, height=520)
