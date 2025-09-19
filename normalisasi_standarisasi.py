import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from tensorflow.python.keras.constraints import min_max_norm

from missing_value import numeric_features

# memuat data yang sudah dibersihkan
try:
    df = pd.read_csv("dataset/Bawang Merah Preprocessed.csv")
    print("Data yang sudah di proses berhasil dimuat")
    print(f"Ukuran Data: {df.shape}")
except FileNotFoundError:
    print("file tidak ditemukan")
    print("harap masukan file data bersih")
    exit()

# numeric feacture
numeric_features = df.select_dtypes(include= ['number']).columns
print(f"\n Diproses (numeric_feacture) {list(numeric_features)}")

# standarisasi data
print("\n Perbandingan Standarisasi")

df_standardized = df.copy()

print("\nData sebelum standarisasi")
print(df[numeric_features].head())
print("\nstatistik deskriptif sebelum standarisasi")
print(df[numeric_features].describe())

# proses standarisasi
scaler = StandardScaler()
df_standardized[numeric_features] = scaler.fit_transform(df[numeric_features])

# show after
print("\nData sesudah standarisasi")
print(df_standardized[numeric_features].head())
print("\nStatistik Deskriptif sesudah")
print(df_standardized[numeric_features].describe())

# visualisasi perbandingan contoh fitur "harga"
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
sns.kdeplot(df['Harga'], fill= True)
plt.title("Sebelum")

plt.subplot(1, 2, 2)
sns.kdeplot(df_standardized['Harga'], fill=True)
plt.title("Sesudah")
plt.suptitle("Distribusi 'Harga'", fontsize=16)
plt.show()


# normalisasi
print("\n\n Perbandingan Normalisasi")

# buat salinan
df_normalized = df.copy()

# before
print("\n Before")
print(df[numeric_features].head())

# proses normalisasi
min_max_scaler = MinMaxScaler()
df_normalized[numeric_features] = min_max_scaler.fit_transform(df[numeric_features])

# after normalisasi
print("\n Data sesudah normalisasi")
print(df_normalized[numeric_features].head())
print("\n Statistik sesudah")
print(df_normalized[numeric_features].describe())

# visualisasi
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['Harga'], fill=True)
plt.title("Sebelum Normalisasi")

plt.subplot(1, 2, 2)
sns.histplot(df_normalized['Harga'], fill=True)
plt.title("Sesudah Normalisasi")
plt.suptitle("Distribusi 'Harga'", fontsize=16)
plt.show()