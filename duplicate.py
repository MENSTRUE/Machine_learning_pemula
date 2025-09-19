import pandas as pd

# memuat data

try:
    df = pd.read_csv("dataset/Bawang Merah preprocessed.csv")
    print("Data yang sudah diproses berhasil dimuat.")
    print(f"Ukuran data awal: {df.shape}")
except FileNotFoundError:
    print("File 'Bawang Merah_preprocessed.csv' tidak ditemukan.")
    print("Harap jalankan skrip '04_data_preprocessing.py' terlebih dahulu.")
    exit()

# menangani duplicate

print("\n Memeriksa data duplikat")
duplicates = df.duplicated()

if duplicates.any():
    print(f"\n Ditemukan {duplicates.sum()} baris duplikat")
    print(df[duplicates])

    # print("\nMenghapus baris duplikat...")
    # df_no_duplicates = df.drop_duplicates()

    # print("DataFrame setelah menghapus duplikat:")
    # print(df_no_duplicates.shape)



else:
    print("\ntidak ada duplicate")

print(f"\n ukuran data akhir: {df.shape}")