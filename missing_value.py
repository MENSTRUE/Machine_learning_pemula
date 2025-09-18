import pandas as pd

# Baris ini harus dijalankan setelah 01_data_loading.py
# Untuk tujuan demonstrasi, kita akan memuat ulang data di sini.
# Dalam alur kerja nyata, Anda akan meneruskan DataFrame 'train'.
train = pd.read_csv("dataset/Bawang Merah.csv")


# --- Penanganan Nilai Hilang ---

# Memeriksa jumlah nilai yang hilang di setiap kolom (missing value)
print("Jumlah Nilai Hilang Sebelum Penanganan:")
missing_values = train.isnull().sum()
nilai_hilang = missing_values[missing_values > 0]
print(nilai_hilang)

# Pisahkan kolom berdasarkan persentase nilai yang hilang
threshold = 0.75 * len(train)
less_missing_cols = missing_values[missing_values < threshold].index
over_missing_cols = missing_values[missing_values >= threshold].index

print("\nKolom dengan nilai hilang < 75%:", list(less_missing_cols))
print("Kolom dengan nilai hilang >= 75%:", list(over_missing_cols))


# --- Mengisi Nilai Hilang untuk Kolom dengan Missing Value < 75% ---

# Mengisi nilai hilang dengan median untuk kolom numerik
# 1. Memilih nama kolom dari DataFrame train yang memiliki tipe data numerik
numeric_features = train[less_missing_cols].select_dtypes(include=['number']).columns

# 2. Hitung nilai median untuk setiap kolom numerik
median_values = train[numeric_features].median()
print("\nMedian yang akan digunakan untuk mengisi data hilang:")
print(median_values)

# 3. Mengisi semua nilai yang hilang (NaN) pada kolom numerik dengan median
train[numeric_features] = train[numeric_features].fillna(median_values)
print("\nNilai hilang pada fitur numerik telah diisi.")

# Mengisi nilai hilang dengan modus untuk kolom kategorikal
kategorial_features = train[less_missing_cols].select_dtypes(include=['object']).columns

for column in kategorial_features:
    modus = train[column].mode()[0]
    train[column] = train[column].fillna(modus)
print("Nilai hilang pada fitur kategorikal telah diisi.")


# --- Menghapus Kolom dengan Missing Value >= 75% ---

# Menghapus kolom yang memiliki terlalu banyak nilai yang hilang
df = train.drop(columns=over_missing_cols)
print(f"\nKolom {list(over_missing_cols)} telah dihapus.")


# --- Verifikasi Akhir ---

# Lakukan pemeriksaan terhadap data yang sudah melewati tahapan verifikasi missing value
final_missing_values = df.isnull().sum()
print("\nJumlah Nilai Hilang Setelah Penanganan:")
print(final_missing_values[final_missing_values > 0])
if final_missing_values.sum() == 0:
    print("Tidak ada nilai hilang yang tersisa dalam DataFrame.")

# Simpan DataFrame yang sudah bersih (opsional)
# df.to_csv("dataset/Bawang Merah_cleaned_missing_values.csv", index=False)

# Menampilkan beberapa baris pertama dari DataFrame yang sudah bersih
print("\nData Setelah Penanganan Nilai Hilang:")
print(df.head())
