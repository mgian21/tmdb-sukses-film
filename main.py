from src.data_loader import load_data
from src.evaluate import evaluasi, grafik_eda
from src.preprocess import bersihkan, siapkan_xy
from src.train import latih_model


def main():
    print("1. Baca data")
    df = load_data()
    print(f"   baris awal: {len(df)}")

    print("2. Bersihkan data")
    df, genre = bersihkan(df)
    print(f"   baris setelah bersih: {len(df)}")
    print(f"   persen film sukses: {df['sukses'].mean() * 100:.1f}%")

    print("3. Grafik EDA")
    grafik_eda(df, genre)

    print("4. Bagi data dan latih model")
    X_tr, X_te, y_tr, y_te, fitur = siapkan_xy(df, genre)
    models = latih_model(X_tr, y_tr)

    print("5. Evaluasi")
    tabel = evaluasi(models, X_te, y_te, y_tr, fitur)
    print(tabel.to_string(index=False))

    print("\nSelesai. Lihat folder outputs/figures dan outputs/tables.")


if __name__ == "__main__":
    main()
