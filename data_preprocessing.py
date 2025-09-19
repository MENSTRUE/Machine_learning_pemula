import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from missing_value import less_missing_cols, over_missing_cols, kategorial_features
from outlier import lower_bound, upper_bound

# memuat data
print("memuat data")
df_awal = pd.read_csv("dataset/Bawang Merah.csv")
print("Data awal berhasil dimuat")
print(f"Ukuran data awal: {df_awal.shape}")

# penanganan nilai hilang
print("penanganan nilai hilang")

missing_values = df_awal.isnull().sum()

threshold = 0.75 * len(df_awal)
less_missing_cols = missing_values[missing_values < threshold].index
over_missing_cols = missing_values[missing_values >= threshold].index

df = df_awal.copy() #menjaga data asli

# menggunakan median untuk mengisi nilai numerik (missing)
numeric_feactures = df[less_missing_cols].select_dtypes(include=['number']).columns
median_values = df[numeric_feactures].median()
df[numeric_feactures] = df[numeric_feactures].fillna(median_values)

# menggunakan modus untuk kolom kategorial (missing)
kategorial_features = df[less_missing_cols].select_dtypes(include=['object']).columns
for column in kategorial_features:
    modus = df[column].mode()[0]
    df[column] = df[column].fillna(modus)

# menghapus kolom yang terlalu banyak nilai hilang
df = df.drop(columns=over_missing_cols)
print("penanganan nilai hilang selesai")
print(f"kolom yang dihapus: {list(over_missing_cols)}")
print(f"ukuran data setelah menangani nilai hilang: {df.shape}")

# penanganan outlier
print("\n memulai penanganan outlier")

numeric_feactures_df = df.select_dtypes(include=['number']).columns

Q1 = df[numeric_feactures_df].quantile(0.25)
Q3 = df[numeric_feactures_df].quantile(0.75)
IQR = Q3 - Q1

# menentukan batas bawah dan atas
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# filter dataframe
condition = ~((df[numeric_feactures_df] < lower_bound) | (df[numeric_feactures_df] > upper_bound)).any(axis=1)
df_clean = df[condition].copy()

print("penanganan outlier selesai")
print(f"jumlah baris yang dihapus: {df.shape[0] - df_clean.shape[0]}")
print(f"ukuran data after: {df_clean.shape}")

# standarisasi fitur numeric

print("/n dsata preprocessing selesai")
print("Data final siap untuk pemodelan")
print(df_clean.head())

# simpan data
df_clean.to_csv("dataset/Bawang Merah Preprocessed.csv", index=False)
print("\n Data yang telah di proses disimpan")