import html
import json
import math

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

# Gaya visual: Opsi A Noir & Emerald (Bioskop Klasik / Retro Prestisius), Playfair Display (88px) & DM Sans
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Playfair+Display:ital,wght@0,600;0,700;1,400&display=swap');

    :root {
        --c-pine: #0B0E0D;
        --c-forest: #141A17;
        --c-paper: #F2EFE9;
        --c-sage: #8F9E96;
        --c-gold: #D4AF37;
        --c-emerald: #10B981;
        --c-rust: #C2410C;
        --border-forest: #222C27;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--c-paper);
        background-color: var(--c-pine);
        overflow-x: hidden;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3.5rem;
        max-width: 1240px;
    }

    h1, h2, h3, .font-playfair {
        font-family: 'Playfair Display', Georgia, serif !important;
        color: var(--c-paper) !important;
        letter-spacing: -0.015em;
    }

    /* Tajuk Utama: Playfair Display 88px di layar utama, tanpa animasi muat halaman (Motion 1) */
    .hero-header {
        border-top: 2px solid var(--c-gold);
        border-bottom: 1px solid var(--border-forest);
        padding: 1.5rem 0 1.35rem 0;
        margin-bottom: 1.5rem;
    }

    .hero-display-title {
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: clamp(3rem, 6.8vw, 88px) !important;
        font-weight: 700 !important;
        line-height: 1.02 !important;
        letter-spacing: -0.02em !important;
        color: var(--c-paper) !important;
        margin: 0 0 0.85rem 0 !important;
    }

    .hero-lead {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.04rem;
        color: var(--c-sage);
        max-width: 780px;
        line-height: 1.55;
        margin: 0 0 1.15rem 0;
    }

    /* Navigasi tautan bagian dengan garis bawah Rust Orange (#C2410C) dan transisi hover pendek (Motion 1) */
    .section-nav {
        display: flex;
        flex-wrap: wrap;
        gap: 1.65rem;
        padding-top: 0.35rem;
    }

    .section-link {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--c-paper) !important;
        text-decoration: none !important;
        border-bottom: 2px solid var(--c-rust);
        padding-bottom: 0.2rem;
        transition: color 0.15s ease, border-color 0.15s ease;
    }

    .section-link:hover {
        color: var(--c-gold) !important;
        border-bottom-color: var(--c-gold);
    }

    /* Pita ringkasan arsip di atas permukaan Dark Forest (#141A17) */
    .archive-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 1.25rem 2.25rem;
        background-color: var(--c-forest);
        border: 1px solid var(--border-forest);
        border-radius: 6px;
        padding: 1.05rem 1.45rem;
        margin-bottom: 2.25rem;
        font-variant-numeric: tabular-nums;
    }

    .strip-item {
        flex: 1 1 170px;
        min-width: 145px;
    }

    .strip-label {
        font-size: 0.79rem;
        color: var(--c-sage);
        margin-bottom: 0.2rem;
        font-weight: 500;
    }

    .strip-val {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--c-paper);
    }

    .strip-val.gold {
        color: var(--c-gold);
    }

    .strip-val.emerald {
        color: var(--c-emerald);
    }

    .strip-sub {
        font-size: 0.78rem;
        color: var(--c-sage);
        margin-top: 0.12rem;
    }

    /* Label penanda "PEMBANDING" khusus berwarna Rust Orange (#C2410C) */
    .badge-pembanding {
        display: inline-block;
        background-color: var(--c-rust);
        color: var(--c-paper);
        font-family: 'DM Sans', sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        padding: 0.18rem 0.55rem;
        border-radius: 3px;
        vertical-align: middle;
        text-transform: uppercase;
    }

    /* Kartu Tiket Warm Paper (#F2EFE9) dengan Stub Rust Orange (#C2410C) */
    .cinema-ticket {
        display: flex;
        flex-direction: row;
        background-color: var(--c-paper);
        color: var(--c-pine);
        border-radius: 6px;
        overflow: hidden;
        border: 1px solid #D5CFC4;
        font-variant-numeric: tabular-nums;
        margin-bottom: 0.85rem;
    }

    .ticket-body {
        flex: 1 1 auto;
        padding: 1.35rem 1.45rem;
        background-color: var(--c-paper);
        color: var(--c-pine);
    }

    .ticket-stub {
        flex: 0 0 160px;
        background-color: var(--c-rust);
        color: var(--c-paper);
        border-left: 2px dashed var(--c-pine);
        padding: 1.35rem 1rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        align-items: center;
        text-align: center;
    }

    .ticket-kicker {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #424A45;
        margin-bottom: 0.25rem;
    }

    .ticket-verdict {
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: 1.55rem;
        font-weight: 700;
        line-height: 1.15;
        color: var(--c-pine) !important;
        margin: 0 0 0.55rem 0;
    }

    .ticket-specs {
        font-size: 0.88rem;
        color: #232926;
        line-height: 1.5;
        margin-bottom: 0.85rem;
    }

    .ticket-divider {
        border-top: 1px dashed #A5ACA8;
        margin: 0.8rem 0;
        padding-top: 0.75rem;
    }

    .ticket-compare-text {
        font-size: 0.84rem;
        color: #232926;
        line-height: 1.45;
        margin-top: 0.35rem;
    }

    .ticket-consensus-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.84rem;
        padding: 0.25rem 0;
        border-bottom: 1px solid #DFDBD2;
        color: var(--c-pine);
    }

    .ticket-consensus-row:last-child {
        border-bottom: none;
    }

    .stub-label {
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--c-paper);
    }

    .stub-prob {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.05;
        color: var(--c-paper);
        margin: 0.4rem 0;
    }

    .stub-verdict-tag {
        background-color: var(--c-pine);
        color: var(--c-gold);
        font-size: 0.73rem;
        font-weight: 700;
        padding: 0.26rem 0.6rem;
        border-radius: 3px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .stub-verdict-tag.hit {
        color: var(--c-emerald);
    }

    /* Panel Tabel Selebar Halaman di atas Latar Dark Forest (#141A17) tanpa geser kiri-kanan */
    .forest-table-section {
        background-color: var(--c-forest);
        border: 1px solid var(--border-forest);
        border-radius: 6px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1.65rem;
        width: 100%;
        box-sizing: border-box;
    }

    .cinema-table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.88rem;
        font-variant-numeric: tabular-nums;
        color: var(--c-paper);
    }

    .cinema-table th {
        text-align: left;
        color: var(--c-sage);
        font-weight: 600;
        font-size: 0.8rem;
        padding: 0.65rem 0.6rem;
        border-bottom: 2px solid var(--border-forest);
        word-wrap: break-word;
        overflow-wrap: break-word;
    }

    .cinema-table td {
        padding: 0.62rem 0.6rem;
        border-bottom: 1px solid #1C2520;
        vertical-align: top;
        word-wrap: break-word;
        overflow-wrap: break-word;
        line-height: 1.4;
    }

    /* Motion 1: Transisi hover pendek pada baris tabel */
    .cinema-table tbody tr {
        transition: background-color 0.14s ease;
    }

    .cinema-table tbody tr:hover {
        background-color: #1C2621;
    }

    .cinema-table .num-col {
        text-align: right;
    }

    /* Sorotan nilai terbaik dengan Antique Gold (#D4AF37) & status Emerald (#10B981) */
    .best-gold {
        color: var(--c-gold);
        font-weight: 700;
    }

    .status-hit-text {
        color: var(--c-emerald);
        font-weight: 700;
    }

    .status-flop-text {
        color: var(--c-sage);
        font-weight: 500;
    }

    .status-incomplete-text {
        color: var(--c-rust);
        font-weight: 600;
    }

    /* Tombol Antique Gold (#D4AF37) & fokus keyboard (Motion 1, R-25, R-32) */
    div.stButton > button[kind="primary"] {
        background-color: var(--c-gold) !important;
        color: var(--c-pine) !important;
        border: 1px solid var(--c-gold) !important;
        border-radius: 4px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 700 !important;
        min-height: 44px !important;
        padding: 0.55rem 1.35rem !important;
        transition: background-color 0.15s ease !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #E2BE47 !important;
        color: var(--c-pine) !important;
    }

    button:focus-visible,
    input:focus-visible,
    a:focus-visible,
    [role="tab"]:focus-visible,
    [data-baseweb="select"] *:focus-visible {
        outline: 2px solid var(--c-gold) !important;
        outline-offset: 2px !important;
    }

    .section-intro {
        color: var(--c-sage);
        font-size: 0.94rem;
        margin-top: -0.3rem;
        margin-bottom: 1.15rem;
        line-height: 1.55;
    }

    .state-box {
        background-color: var(--c-forest);
        border: 1px solid var(--border-forest);
        border-radius: 6px;
        padding: 0.9rem 1.15rem;
        color: var(--c-sage);
        font-size: 0.89rem;
        line-height: 1.5;
        margin-bottom: 0.95rem;
    }

    .state-box.pending {
        border-color: var(--c-gold);
        color: var(--c-paper);
    }

    @media (max-width: 900px) {
        .cinema-ticket {
            flex-direction: column;
        }
        .ticket-stub {
            flex: 0 0 auto;
            border-left: none;
            border-top: 2px dashed var(--c-pine);
            flex-direction: row;
            justify-content: space-between;
            padding: 0.9rem 1.25rem;
        }
        .cinema-table {
            font-size: 0.78rem;
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


def _format_usd_ringkas(nilai):
    """Format nilai dolar agar ringkas dan tidak memicu geser kiri-kanan pada tabel."""
    if pd.isna(nilai):
        return "-"
    nilai = float(nilai)
    tanda = "-" if nilai < 0 else ""
    abs_n = abs(nilai)
    if abs_n == 0:
        return "$0"
    if abs_n >= 1_000_000_000:
        return f"{tanda}${abs_n / 1_000_000_000:.2f} M"
    if abs_n >= 1_000_000:
        return f"{tanda}${abs_n / 1_000_000:.1f} Jt"
    if abs_n >= 1_000:
        return f"{tanda}${abs_n / 1_000:.0f} Rb"
    return f"{tanda}${abs_n:.0f}"


@st.cache_data
def muat_data():
    df_mentah = pd.read_csv(DATA_PATH)

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
            "tn_count": tn_base,
            "Deteksi Film Rugi": f"{tn_base} / {total_rugi_uji} ({tn_base / total_rugi_uji * 100:.1f}%)",
            "is_baseline": True,
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
                "tn_count": tn,
                "Deteksi Film Rugi": f"{tn} / {total_rugi_uji} ({tn / total_rugi_uji * 100:.1f}%)",
                "is_baseline": False,
            }
        )
    return models, X_te, y_te, fitur, pd.DataFrame(baris)


def render_tabel_evaluasi_html(df_tabel):
    """Render tabel perbandingan model selebar halaman di atas latar Dark Forest (#141A17)."""
    df_ml = df_tabel[~df_tabel["is_baseline"]]
    maks_acc = df_ml["Accuracy"].max()
    maks_prec = df_ml["Precision"].max()
    maks_rec = df_ml["Recall"].max()
    maks_f1 = df_ml["F1"].max()
    maks_tn = df_ml["tn_count"].max()

    baris_html = []
    for _, row in df_tabel.iterrows():
        nama = html.escape(str(row["Model"]))
        if row["is_baseline"]:
            label_model = (
                f'{nama} &nbsp;<span class="badge-pembanding">PEMBANDING</span>'
            )
        else:
            label_model = f"<strong>{nama}</strong>"

        def _sel(nilai, maks, is_base):
            teks = f"{nilai:.3f}"
            if (not is_base) and abs(nilai - maks) < 1e-6:
                return f'<td class="num-col"><span class="best-gold">{teks} ★</span></td>'
            return f'<td class="num-col">{teks}</td>'

        tn_teks = html.escape(str(row["Deteksi Film Rugi"]))
        if (not row["is_baseline"]) and row["tn_count"] == maks_tn:
            sel_tn = f'<td class="num-col"><span class="best-gold">{tn_teks} ★</span></td>'
        else:
            sel_tn = f'<td class="num-col">{tn_teks}</td>'

        baris_html.append(
            f"<tr>"
            f"<td>{label_model}</td>"
            f'{_sel(row["Accuracy"], maks_acc, row["is_baseline"])}'
            f'{_sel(row["Precision"], maks_prec, row["is_baseline"])}'
            f'{_sel(row["Recall"], maks_rec, row["is_baseline"])}'
            f'{_sel(row["F1"], maks_f1, row["is_baseline"])}'
            f"{sel_tn}"
            f"</tr>"
        )

    return (
        '<div class="forest-table-section">'
        '<table class="cinema-table" aria-label="Tabel Perbandingan Model Klasifikasi">'
        "<colgroup>"
        '<col style="width: 30%;">'
        '<col style="width: 13%;">'
        '<col style="width: 13%;">'
        '<col style="width: 13%;">'
        '<col style="width: 13%;">'
        '<col style="width: 18%;">'
        "</colgroup>"
        "<thead><tr>"
        "<th>Algoritma / Model</th>"
        '<th class="num-col">Accuracy</th>'
        '<th class="num-col">Precision</th>'
        '<th class="num-col">Recall</th>'
        '<th class="num-col">Skor F1</th>'
        '<th class="num-col">Deteksi Film Rugi</th>'
        "</tr></thead>"
        f'<tbody>{"".join(baris_html)}</tbody>'
        "</table>"
        "</div>"
    )


def render_tabel_arsip_html(df_halaman):
    """Render tabel arsip film selebar halaman di atas latar Dark Forest (#141A17) tanpa horizontal scroll."""
    baris_html = []
    for _, row in df_halaman.iterrows():
        judul = html.escape(str(row["title"]))
        tahun = (
            str(int(row["tahun_rilis"]))
            if pd.notna(row["tahun_rilis"]) and row["tahun_rilis"] > 0
            else "-"
        )
        genre_str = html.escape(str(row["genre_teks"]))
        anggaran = _format_usd_ringkas(row["budget"])
        pendapatan = _format_usd_ringkas(row["revenue"])
        selisih = _format_usd_ringkas(row["selisih_usd"])
        durasi = (
            f'{int(row["runtime"])} mnt'
            if pd.notna(row["runtime"]) and row["runtime"] > 0
            else "-"
        )
        status = str(row["status_arsip"])
        if status == "Untung (Hit)":
            kelas_status = "status-hit-text"
        elif status == "Rugi (Flop)":
            kelas_status = "status-flop-text"
        else:
            kelas_status = "status-incomplete-text"

        baris_html.append(
            f"<tr>"
            f"<td><strong>{judul}</strong></td>"
            f"<td>{tahun}</td>"
            f"<td>{genre_str}</td>"
            f'<td class="num-col">{anggaran}</td>'
            f'<td class="num-col">{pendapatan}</td>'
            f'<td class="num-col">{selisih}</td>'
            f'<td class="num-col">{durasi}</td>'
            f'<td><span class="{kelas_status}">{html.escape(status)}</span></td>'
            f"</tr>"
        )

    return (
        '<div class="forest-table-section">'
        '<table class="cinema-table" aria-label="Tabel Arsip Film TMDB 5000">'
        "<colgroup>"
        '<col style="width: 22%;">'
        '<col style="width: 7%;">'
        '<col style="width: 23%;">'
        '<col style="width: 11%;">'
        '<col style="width: 11%;">'
        '<col style="width: 11%;">'
        '<col style="width: 6%;">'
        '<col style="width: 9%;">'
        "</colgroup>"
        "<thead><tr>"
        "<th>Judul Film</th>"
        "<th>Tahun</th>"
        "<th>Genre</th>"
        '<th class="num-col">Anggaran</th>'
        '<th class="num-col">Pendapatan</th>'
        '<th class="num-col">Selisih</th>'
        '<th class="num-col">Durasi</th>'
        "<th>Status</th>"
        "</tr></thead>"
        f'<tbody>{"".join(baris_html)}</tbody>'
        "</table>"
        "</div>"
    )


def gambar_confusion_matrix(model, X_te, y_te, nama_model):
    """Buat plot Confusion Matrix dengan kontras tinggi sesuai palet Noir & Emerald (R-25)."""
    y_pred = model.predict(X_te)
    cm = confusion_matrix(y_te, y_pred)

    cmap_noir = mcolors.LinearSegmentedColormap.from_list(
        "noir_gold", ["#0B0E0D", "#6E5A1F", "#D4AF37"]
    )
    fig, ax = plt.subplots(figsize=(4.6, 3.4), facecolor="#141A17")
    ax.set_facecolor("#141A17")
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap_noir)

    labels = ["Rugi (Flop)", "Untung (Hit)"]
    ax.set_xticks([0, 1], labels=labels, color="#F2EFE9", fontsize=9.5)
    ax.set_yticks([0, 1], labels=labels, color="#F2EFE9", fontsize=9.5)
    ax.set_xlabel("Prediksi Model", color="#8F9E96", fontsize=9, labelpad=8)
    ax.set_ylabel("Kenyataan Box Office", color="#8F9E96", fontsize=9, labelpad=8)
    ax.set_title(
        f"Matriks Evaluasi: {nama_model}",
        color="#F2EFE9",
        fontsize=10.5,
        fontweight="bold",
        pad=10,
    )

    for spine in ax.spines.values():
        spine.set_edgecolor("#222C27")

    batas = (cm.max() + cm.min()) / 2.0
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            nilai = cm[i, j]
            persen = (nilai / total) * 100 if total > 0 else 0.0
            warna_teks = "#0B0E0D" if nilai > batas else "#F2EFE9"
            ax.text(
                j,
                i,
                f"{nilai}\n({persen:.1f}%)",
                ha="center",
                va="center",
                color=warna_teks,
                fontsize=9.5,
                fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#8F9E96", labelcolor="#8F9E96", labelsize=8)
    cbar.outline.set_edgecolor("#222C27")
    fig.tight_layout()
    return fig


def gambar_batang_horizontal(
    series_data, warna_batang, format_nilai="{:.0f}", tinggi=3.2
):
    """Buat grafik batang horizontal terurut dari nilai terbesar di atas di atas kanvas Dark Forest."""
    s = series_data.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(5.2, tinggi), facecolor="#141A17")
    ax.set_facecolor("#141A17")

    if isinstance(warna_batang, dict):
        warna_list = [warna_batang.get(k, "#D4AF37") for k in s.index]
    else:
        warna_list = warna_batang

    bars = ax.barh(s.index, s.values, color=warna_list, height=0.6)
    ax.tick_params(axis="y", colors="#F2EFE9", labelsize=9)
    ax.tick_params(axis="x", colors="#8F9E96", labelsize=8.5)
    ax.xaxis.grid(True, linestyle="--", alpha=0.22, color="#8F9E96")
    ax.set_axisbelow(True)

    maks = float(s.max()) if len(s) > 0 else 1.0
    ax.set_xlim(0, maks * 1.18)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_edgecolor("#222C27")

    for bar in bars:
        lebar = bar.get_width()
        ax.text(
            lebar + maks * 0.02,
            bar.get_y() + bar.get_height() / 2,
            format_nilai.format(lebar),
            va="center",
            ha="left",
            color="#F2EFE9",
            fontsize=8.5,
            fontweight="bold",
        )

    fig.tight_layout()
    return fig


# ---------- Tajuk Utama (Playfair Display 88px & Tautan Bergaris Bawah Rust Orange) ----------
st.markdown(
    """
    <header class="hero-header">
        <h1 class="hero-display-title">Proyeksi Box Office.</h1>
        <p class="hero-lead">
            Analisis kelayakan komersial produksi film (pendapatan melampaui anggaran) berdasarkan arsip historis
            TMDB 5000 menggunakan Logistic Regression, Decision Tree, dan Random Forest.
        </p>
        <nav class="section-nav" aria-label="Navigasi Bagian Halaman">
            <a class="section-link" href="#studio-proyeksi">Studio Proyeksi &amp; Tiket</a>
            <a class="section-link" href="#evaluasi-model">Evaluasi Model Klasifikasi</a>
            <a class="section-link" href="#arsip-tmdb">Arsip Film TMDB 5000</a>
        </nav>
    </header>
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

terbaik_acc = (
    tabel[~tabel["is_baseline"]].sort_values("Accuracy", ascending=False).iloc[0]
)
tahun_min_data = int(df["tahun"].min())
tahun_maks_data = int(df["tahun"].max())

# ---------- Ringkasan Arsip ----------
st.markdown(
    f"""
    <div class="archive-strip" role="region" aria-label="Ringkasan Arsip TMDB">
        <div class="strip-item">
            <div class="strip-label">Arsip Bersih Terverifikasi</div>
            <div class="strip-val">{len(df):,} Film</div>
            <div class="strip-sub">Dari {len(df_mentah):,} total baris CSV ({tahun_min_data}-{tahun_maks_data})</div>
        </div>
        <div class="strip-item">
            <div class="strip-label">Tingkat Balik Modal Historis</div>
            <div class="strip-val emerald">{df['sukses'].mean() * 100:.1f}%</div>
            <div class="strip-sub">{int(df['sukses'].sum()):,} Hit vs {int((df['sukses'] == 0).sum()):,} Flop</div>
        </div>
        <div class="strip-item">
            <div class="strip-label">Akurasi Model Tertinggi</div>
            <div class="strip-val gold">{terbaik_acc['Model']}</div>
            <div class="strip-sub">Accuracy {terbaik_acc['Accuracy'] * 100:.1f}% | F1 {terbaik_acc['F1']:.3f}</div>
        </div>
        <div class="strip-item">
            <div class="strip-label">Sampel Pengujian</div>
            <div class="strip-val">{len(X_te):,} Film</div>
            <div class="strip-sub">20% data uji terstratifikasi</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# BAGIAN 1: Studio Proyeksi & Tiket Bioskop (Warm Paper + Stub Rust Orange)
# ==============================================================================
st.markdown('<div id="studio-proyeksi"></div>', unsafe_allow_html=True)
st.subheader("Studio Proyeksi Box Office")
st.markdown(
    '<p class="section-intro">'
    "Simulasikan rencana anggaran, durasi tayang, tahun rilis, dan kombinasi genre untuk menguji peluang balik modal produksi film Anda."
    "</p>",
    unsafe_allow_html=True,
)

kol_input, kol_tiket = st.columns([1.12, 0.88], gap="large")

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
            help="Pilih algoritma penentu utama pada tiket. Konsensus ketiga model tetap ditampilkan di dalam tiket.",
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
            "Tekan tombol <strong>Hitung Proyeksi Box Office</strong> untuk memperbarui hasil tiket di bawah."
            "</div>",
            unsafe_allow_html=True,
        )

    peluang_utama = hasil["peluang_utama"]
    is_hit = peluang_utama >= 0.5
    judul_keputusan = (
        "BOX OFFICE HIT (UNTUNG)" if is_hit else "BOX OFFICE FLOP (RUGI)"
    )
    tag_stub = "PROYEKSI HIT" if is_hit else "PROYEKSI FLOP"
    kelas_tag_stub = "stub-verdict-tag hit" if is_hit else "stub-verdict-tag"
    label_genre = (
        html.escape(", ".join(hasil["genre"]))
        if hasil["genre"]
        else "Tanpa Genre Spesifik"
    )
    nama_model_aman = html.escape(str(hasil["model_utama"]))

    baris_konsensus = "".join(
        [
            f'<div class="ticket-consensus-row">'
            f'<span>{html.escape(nama_m)}{" (Utama)" if nama_m == hasil["model_utama"] else ""}</span>'
            f"<strong>{prob * 100:.1f}% untung</strong>"
            f"</div>"
            for nama_m, prob in hasil["semua_peluang"].items()
        ]
    )

    st.markdown(
        f"""
        <div class="cinema-ticket" role="status" aria-live="polite">
            <div class="ticket-body">
                <div class="ticket-kicker">Tiket Evaluasi Produksi &bull; Rilis {hasil['tahun']}</div>
                <div class="ticket-verdict">{judul_keputusan}</div>
                <div class="ticket-specs">
                    Anggaran: <strong>USD {hasil['budget_juta']} Juta</strong> &nbsp;|&nbsp;
                    Durasi: <strong>{hasil['runtime']} menit</strong><br>
                    Genre: <strong>{label_genre}</strong>
                </div>
                <div class="ticket-divider">
                    <span class="badge-pembanding">PEMBANDING</span>
                    <div class="ticket-compare-text">
                        Pada <strong>{hasil['jumlah_film_genre']:,} film sejenis</strong> di arsip TMDB,
                        median anggaran adalah <strong>USD {hasil['median_budget_genre']:.1f} Juta</strong>
                        dengan tingkat balik modal historis <strong>{hasil['rata_sukses_genre']:.1f}%</strong>.
                    </div>
                </div>
                <div class="ticket-divider">
                    <div class="ticket-kicker">Konsensus Tiga Algoritma</div>
                    {baris_konsensus}
                </div>
            </div>
            <div class="ticket-stub">
                <div class="stub-label">Peluang Untung<br>({nama_model_aman})</div>
                <div class="stub-prob">{peluang_utama * 100:.1f}%</div>
                <div class="{kelas_tag_stub}">{tag_stub}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ==============================================================================
# BAGIAN 2: Evaluasi & Perbandingan Model Klasifikasi
# ==============================================================================
st.markdown('<div id="evaluasi-model"></div>', unsafe_allow_html=True)
st.subheader("Evaluasi & Perbandingan Model Klasifikasi")
st.markdown(
    '<p class="section-intro">'
    "Perbandingan kinerja Logistic Regression, Decision Tree, dan Random Forest terhadap acuan tebakan mayoritas pada 646 film data uji (488 Hit dan 158 Flop)."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown(render_tabel_evaluasi_html(tabel), unsafe_allow_html=True)

st.markdown(
    '<div class="state-box">'
    "<strong>Mengapa Baseline memiliki Recall 1.000?</strong> "
    "Karena 75,5% film pada arsip bersih merupakan film untung, <em>Baseline (tebak mayoritas)</em> menebak seluruh 646 film uji sebagai film untung tanpa terkecuali, "
    "sehingga gagal mendeteksi satu pun film yang merugi (0 / 158 film rugi). "
    "<strong>Random Forest</strong> dan <strong>Decision Tree</strong> mencapai Precision lebih tinggi sekaligus mendeteksi puluhan film yang berpotensi merugi."
    "</div>",
    unsafe_allow_html=True,
)

diag_kiri, diag_kanan = st.columns([0.95, 1.05], gap="large")

with diag_kiri:
    st.subheader("Diagnosis Kesalahan Prediksi (Confusion Matrix)")
    pilih_model = st.selectbox(
        "Pilih model untuk melihat rincian tebakan benar vs meleset:",
        list(models.keys()),
        index=2,
    )
    fig_cm = gambar_confusion_matrix(models[pilih_model], X_te, y_te, pilih_model)
    st.pyplot(fig_cm, width="stretch")
    plt.close(fig_cm)

with diag_kanan:
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
        penting, warna_batang="#D4AF37", format_nilai="{:.3f}", tinggi=3.4
    )
    st.pyplot(fig_imp, width="stretch")
    plt.close(fig_imp)

st.divider()

# ==============================================================================
# BAGIAN 3: Eksplorasi & Tabel Arsip TMDB 5000 Selebar Halaman
# ==============================================================================
st.markdown('<div id="arsip-tmdb"></div>', unsafe_allow_html=True)
st.subheader("Eksplorasi Arsip TMDB 5000")
st.markdown(
    '<p class="section-intro">'
    "Proporsi keberhasilan komersial lintas genre serta direktori lengkap film dari arsip TMDB 5000."
    "</p>",
    unsafe_allow_html=True,
)

eda_kiri, eda_kanan = st.columns([0.9, 1.1], gap="large")

with eda_kiri:
    st.markdown("**Berapa banyak film di arsip bersih yang berhasil balik modal?**")
    hitung_status = (
        df["sukses"].map({1: "Untung (Hit)", 0: "Rugi (Flop)"}).value_counts()
    )
    fig_status = gambar_batang_horizontal(
        hitung_status,
        warna_batang={"Untung (Hit)": "#10B981", "Rugi (Flop)": "#C2410C"},
        format_nilai="{:,.0f} film",
        tinggi=2.2,
    )
    st.pyplot(fig_status, width="stretch")
    plt.close(fig_status)

with eda_kanan:
    st.markdown("**Genre apa yang paling banyak diproduksi dalam arsip TMDB?**")
    per_genre = df[["genre_" + g for g in genre]].sum()
    per_genre.index = genre
    fig_genre = gambar_batang_horizontal(
        per_genre.sort_values(ascending=False).head(10),
        warna_batang="#8F9E96",
        format_nilai="{:,.0f}",
        tinggi=2.8,
    )
    st.pyplot(fig_genre, width="stretch")
    plt.close(fig_genre)

st.markdown("#### Direktori Film Arsip TMDB")

mode_arsip = st.radio(
    "Cakupan Data Arsip",
    options=[
        f"Arsip Bersih Model ({len(df):,} film dengan anggaran & pendapatan > $0)",
        f"Seluruh Data Mentah CSV ({len(df_mentah):,} film, termasuk data $0 / belum tercatat)",
    ],
    index=0,
    horizontal=True,
)

pakai_mentah = mode_arsip.startswith("Seluruh Data Mentah")
df_sumber = df_mentah if pakai_mentah else df

f_col1, f_col2, f_col3, f_col4 = st.columns([1.3, 1.0, 1.0, 0.7])
with f_col1:
    cari_judul = st.text_input(
        "Cari judul film",
        value="",
        placeholder="Misal: Avatar, Spider-Man, Batman...",
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
with f_col4:
    baris_per_hal = st.selectbox(
        "Baris per halaman",
        options=[25, 50, 100],
        index=1,
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
        f"Hapus kata kunci di kolom pencarian atau kembalikan filter ke 'Semua' untuk menampilkan kembali data."
        f"</div>",
        unsafe_allow_html=True,
    )
else:
    total_baris = len(df_tampil)
    total_halaman = max(1, math.ceil(total_baris / baris_per_hal))

    nav_kiri, nav_kanan = st.columns([0.72, 0.28])
    with nav_kanan:
        halaman_aktif = st.number_input(
            f"Halaman (dari {total_halaman})",
            min_value=1,
            max_value=total_halaman,
            value=1,
            step=1,
        )
    idx_awal = (halaman_aktif - 1) * baris_per_hal
    idx_akhir = min(idx_awal + baris_per_hal, total_baris)
    with nav_kiri:
        st.caption(
            f"Menampilkan baris {idx_awal + 1:,} sampai {idx_akhir:,} dari total {total_baris:,} film yang cocok "
            f"(Cakupan: {len(df_sumber):,} film | Jt = Juta USD, M = Miliar USD)."
        )

    potongan_df = df_tampil.iloc[idx_awal:idx_akhir]
    st.markdown(render_tabel_arsip_html(potongan_df), unsafe_allow_html=True)
