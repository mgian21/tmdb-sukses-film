import matplotlib

matplotlib.use("Agg")  # simpan gambar tanpa membuka jendela
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import FIG_DIR, TABLE_DIR


def _simpan(nama):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / nama, dpi=150)
    plt.close()


def grafik_eda(df, genre):
    df["sukses"].map({1: "Sukses", 0: "Tidak sukses"}).value_counts().plot.bar(
        color=["#4C78A8", "#E45756"], title="Sukses vs Tidak Sukses"
    )
    _simpan("eda_label.png")

    df[["genre_" + g for g in genre]].sum().rename(
        lambda n: n.replace("genre_", "")
    ).sort_values().plot.barh(title="Jumlah Film per Genre", figsize=(7, 6))
    _simpan("eda_genre.png")

    (df["budget"] / 1e6).plot.hist(bins=40, title="Distribusi Anggaran (juta USD)")
    _simpan("eda_budget.png")


def evaluasi(models, X_te, y_te, y_tr, fitur):
    baris = []

    # Baseline: selalu tebak kelas mayoritas
    mayoritas = int(y_tr.mode()[0])
    p = [mayoritas] * len(y_te)
    baris.append(
        {
            "Model": "Baseline (tebak mayoritas)",
            "Accuracy": accuracy_score(y_te, p),
            "Precision": precision_score(y_te, p, zero_division=0),
            "Recall": recall_score(y_te, p, zero_division=0),
            "F1": f1_score(y_te, p, zero_division=0),
        }
    )

    for nama, model in models.items():
        p = model.predict(X_te)
        baris.append(
            {
                "Model": nama,
                "Accuracy": accuracy_score(y_te, p),
                "Precision": precision_score(y_te, p),
                "Recall": recall_score(y_te, p),
                "F1": f1_score(y_te, p),
            }
        )
        ConfusionMatrixDisplay.from_estimator(
            model,
            X_te,
            y_te,
            display_labels=["Tidak sukses", "Sukses"],
            cmap="Blues",
        )
        plt.title(f"Confusion Matrix - {nama}")
        _simpan("cm_" + nama.lower().replace(" ", "_") + ".png")

    tabel = pd.DataFrame(baris).round(3)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tabel.to_csv(TABLE_DIR / "hasil_perbandingan.csv", index=False)

    # Fitur terpenting Random Forest
    penting = pd.Series(
        models["Random Forest"].feature_importances_, index=fitur
    ).sort_values()
    penting.tail(10).plot.barh(title="10 Fitur Terpenting (Random Forest)")
    _simpan("feature_importance.png")

    return tabel
