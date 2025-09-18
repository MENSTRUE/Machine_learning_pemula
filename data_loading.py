import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Memuat dataset dari file CSV
train = pd.read_csv("dataset/Bawang Merah.csv")

# Menampilkan lima baris pertama dari dataset
print("Data Awal:")
print(train.head())

# Mengidentifikasi informasi dataset
# Menampilkan ringkasan informasi dari dataset, seperti tipe data dan nilai non-null
print("\nInformasi Dataset:")
train.info()

# Analisis deskriptif
# Menampilkan statistik deskriptif dari dataset untuk semua kolom
print("\nStatistik Deskriptif:")
desc = train.describe(include="all")
print(desc)
    