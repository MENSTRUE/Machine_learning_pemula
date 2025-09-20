import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.pylabtools import figsize
from sklearn.preprocessing import LabelEncoder
from sympy.abc import alpha

from konversi_tipedata import category_feacture, label_encoder

# persiapan data
try:
    df_processed = pd.read_csv("dataset/Bawang Merah preprocessed.csv")
    print("--- Data Awal untuk Analisis (Bawang Merah_preprocessed.csv) ---")
    print(df_processed.head())
except FileNotFoundError:
    print("--- PENTING ---")
    print("File 'dataset/Bawang Merah_preprocessed.csv' tidak ditemukan.")
    print("Silakan jalankan skrip data_preprocessing.py terlebih dahulu untuk membuatnya.")
    # Keluar dari skrip jika file tidak ada
    exit()

print("\n" + "="*50 + "\n")

# label encoding

print("menerapkan label encoding")
df_lencoder = df_processed.copy()

categorical_feactures = df_lencoder.select_dtypes(include=['object']).columns

label_encoder = LabelEncoder()
for col in categorical_feactures:
    df_lencoder[col] = label_encoder.fit_transform(df_lencoder[col])

print('after label encoding')
print(df_lencoder.head())
print("\n" + "="*50 + "\n")

# verifikasi data

# menghitung missing values (perkolom)
missing_values = df_processed.isnull().sum()
missing_percentage = (missing_values / len(df_processed)) * 100

missing_data = pd.DataFrame({
    'Missing values': missing_values,
    'Percentage': missing_percentage
}).sort_values(by='Missing values', ascending=False)

missing_data[missing_data['Missing values'] > 0]


# menampilkan hanya kolom yang memiliki missing value
print('pemeriksaan missing value')
if missing_data[missing_data['Missing values'] > 0].empty:
    print('tidak ditemukan missing value. data sudah bersih')
else:
    print(missing_data[missing_data['missing values'] >0])
print("\n" + "="*50 + "\n")

# analisis distribusi data (histogram keseluruhan)
print('menampilkan histogram untuk setiap kolom')

numeric_df = df_lencoder.select_dtypes(include=['number'])
num_vars = numeric_df.shape[1]

# menentukan jumlah baris dan kolom (gris subplot)
n_cols = 3
n_rows = -(-num_vars // n_cols)

# membuat subplot
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))

# flaten axes array untuk memudahkan iterasi jika diperlukan
axes = axes.flatten()

# plot setiap variable
for i, column in enumerate(numeric_df.columns):
    numeric_df[column].hist(ax=axes[i], bins=20, edgecolor="black")
    axes[i].set_title(f'Distribusi kolom {column}')
    axes[i].set_xlabel('value')
    axes[i].set_ylabel('frequency')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# analisis distribusi kolom spesifik
print("\n" + "="*50 + "\n")
print("analisi distribusi kolom spesifik")


columns_to_plot = ['Komoditas', 'Tgl', 'Harga']

# Exda
# visualisasi distribusi data untuk beberapa kolom
plt.figure(figsize=(15,5))
for i, column in enumerate(columns_to_plot, 1):
    if column in df_lencoder.columns: # memastikan kolom yang di plot ada
        plt.subplot(1, 3, i)
        sns.histplot(df_lencoder[column], kde=True, bins=30)
        plt.title(f'Distribusi {column}')

plt.tight_layout()
plt.show()

# korelasi (heatmap)
print("\n" + "="*50 + "\n")
print("analisis korelasi (heatmap)")
plt.figure(figsize=(12, 10))
correlation_matrix = df_lencoder.corr()

sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("heatmap correlation")
plt.show()


# analisi korelasi (variable target)
print("\n" + "="*50 + "\n")
print("analisis korelasi")

target_corr = numeric_df.corr()['Harga']
target_corr_sorted = target_corr.abs().sort_values(ascending=False)

plt.figure(figsize=(10, 8))
target_corr_sorted.plot(kind='bar')
plt.title('korelasi fitur lain terhadap harga')
plt.xlabel('Variable')
plt.ylabel('Koefisien korelasi (absolut)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()