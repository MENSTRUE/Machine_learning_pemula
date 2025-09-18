import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Kode ini mengasumsikan Anda telah menjalankan skrip sebelumnya.
# Kita akan memuat data yang "dibersihkan" dari nilai hilang jika ada,
# jika tidak, kita akan mengulangi langkah-langkah pembersihan di sini secara singkat
# agar skrip ini dapat berjalan sendiri.

try:
    # Coba muat data yang sudah bersih dari langkah sebelumnya
    df = pd.read_csv("dataset/Bawang Merah_cleaned_missing_values.csv")
except FileNotFoundError:
    print("File yang telah dibersihkan tidak ditemukan. Menjalankan pembersihan nilai hilang lagi...")
    train = pd.read_csv("dataset/Bawang Merah.csv")
    threshold = 0.75 * len(train)
    missing_values = train.isnull().sum()
    less_missing_cols = missing_values[missing_values < threshold].index
    over_missing_cols = missing_values[missing_values >= threshold].index

    numeric_features = train[less_missing_cols].select_dtypes(include=['number']).columns
    train[numeric_features] = train[numeric_features].fillna(train[numeric_features].median())

    kategorial_features = train[less_missing_cols].select_dtypes(include=['object']).columns
    for column in kategorial_features:
        train[column] = train[column].fillna(train[column].mode()[0])

    df = train.drop(columns=over_missing_cols)
    print("Pembersihan nilai hilang selesai.")

# --- Visualisasi Outlier Sebelum Penanganan ---

# Dapatkan daftar kolom numerik dari DataFrame yang sudah bersih
numeric_features_df = df.select_dtypes(include=['number']).columns

print("\nMenampilkan box plot untuk setiap fitur numerik (SEBELUM penanganan outlier):")
for feature in numeric_features_df:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[feature])
    plt.title(f'Box plot of {feature} - SEBELUM')
    plt.show()

# --- Mengidentifikasi dan Menghapus Outliers Menggunakan Metode IQR ---

# 1. Menghitung IQR, Q1, dan Q3
Q1 = df[numeric_features_df].quantile(0.25)
Q3 = df[numeric_features_df].quantile(0.75)
IQR = Q3 - Q1
print("\nIQR untuk setiap fitur numerik:")
print(IQR)

# 2. Menentukan batas bawah dan batas atas
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter dataframe untuk hanya menyimpan baris yang tidak mengandung outliers
# Kondisi ini akan bernilai True untuk baris yang tidak memiliki outlier pada kolom numerik manapun
condition = ~((df[numeric_features_df] < lower_bound) | (df[numeric_features_df] > upper_bound)).any(axis=1)
df_no_outliers = df[condition].copy()

print(f"\nUkuran DataFrame sebelum menghapus outliers: {df.shape}")
print(f"Ukuran DataFrame setelah menghapus outliers: {df_no_outliers.shape}")
print(f"Jumlah baris yang dihapus: {df.shape[0] - df_no_outliers.shape[0]}")

# --- Visualisasi Outlier Setelah Penanganan ---

print("\nMenampilkan box plot untuk setiap fitur numerik (SETELAH penanganan outlier):")
for feature in numeric_features_df:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df_no_outliers[feature])
    plt.title(f'Box plot of {feature} - SETELAH')
    plt.show()

# pro tips yang ada di kode asli
# median = df['column_name'].median()
# df['column_name'] = df['column_name'].apply(lambda x: median if x < (Q1 - 1.5 IQR) or x > (Q3 + 1.5 IQR) else x)
#
# # mengganti outliers dengan nilai batas terdekat
# from tensorflow.python.ops.gen_array_ops import lower_bound # Impor ini sebenarnya tidak diperlukan untuk logika ini
# df['column_name'] = df['column_name'].apply(lambda x: (Q1 - 1.5 IQR) if x < (Q1 - 1.5 IQR) else (Q3 + 1.5 IQR) if x > (Q3 + 1.5 * IQR) else x)

# Menampilkan beberapa baris pertama dari DataFrame final
print("\nData Final Setelah Penanganan Outlier:")
print(df_no_outliers.head())
