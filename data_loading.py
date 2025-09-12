import pandas as pd

train = pd.read_csv("dataset/Bawang Merah.csv")
train.head()
print(train.head())

# Menampilkan ringkasan informasi dari dataset
print(train.info())


# Menampilkan statistik deskriptif dari dataset
desc = train.describe(include="all")
print(desc)

# memeriksa jumlah nilai yang hilang di setiap kolom
missing_values = train.isnull().sum()
nilai_hilang = missing_values[missing_values > 0]
print(nilai_hilang)