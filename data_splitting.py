import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from EDA_ExDA import categorical_feactures, label_encoder

# persiapan data
try:
    df_processed = pd.read_csv("dataset/Bawang Merah Preprocessed.csv")
except FileNotFoundError:
    print("--- PENTING ---")
    print("File 'dataset/Bawang Merah_preprocessed.csv' tidak ditemukan.")
    print("Silakan jalankan skrip 04_data_preprocessing.py terlebih dahulu untuk membuatnya.")
    exit()

# label encoding
df_lencoder = df_processed.copy()
categorical_feactures = df_lencoder.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for col in categorical_feactures:
    df_lencoder[col] = label_encoder.fit_transform(df_lencoder[col])

print("Data siap di split")
print(df_lencoder.head())
print("\n" + "="*50 + "\n")

# memisahkan fitur
x = df_lencoder.drop(columns=['Harga'])
y = df_lencoder['Harga']

print("Fitur X")
print(x.head())
print("Traget Y")
print(y.head())
print("\n" + "="*50 + "\n")

# membagi dataset (data splitting)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

print("Data splitting selesai")
print("Data telah dibagi menjadi data latih (train) dan data uji (test).")
print("\n" + "="*50 + "\n")

# verifikasi hasil split
print("--- Ukuran Hasil Pembagian Data ---")
print(f"jumlah data asli: {len(x)}")
print(f"jumlah data latih: {len(x_train)}")
print(f"Jumlah data uji: {len(x_test)}")
