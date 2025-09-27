import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

x, y = make_classification(n_samples=1000, n_features=15, n_informative=10, n_redundant=2, n_clusters_per_class=1, weights=[0.9], flip_y=0, random_state=42)

# mengubah nilai acak yang tersimpan dengan tipe data array menjadi dataframe
# menyusun dataset menjadi dataframe untuk kemudahan
df = pd.DataFrame(x, columns=[f'Fitur_{i}' for i in range(1, 16)])
df['Target'] = y

# misalkan kita punya beberapa fitur kategorikal (simulasi fitur kategorikal)
df['Fitur_12'] = np.random.choice(['A', 'B', 'C'], size=1000)
df['Fitur_13'] = np.random.choice(['X', 'Y', 'Z'], size=1000)

print(df)

# meisahkan fitur dan target
x = df.drop('Target',axis=1)
y = df['Target']

# memastikan prosisi data yang sudah ada / melihat distribusi kelas
print("Distribusi kelas sebelum smote:", Counter(y))

# ------------------- Embedded Methods -------------------
# Menggunakan Random Forest untuk mendapatkan fitur penting
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
x_integer = x.drop(['Fitur_12', 'Fitur_13'], axis=1)
rf_model.fit(x_integer, y)

# mendapatkan fitur penting
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# menentukan ambang batas untuk fitur penting
threshold = 0.05 #misalnya ambang batas 5%
important_features_indices = [i for i in range(len(importances)) if importances[i] >= threshold]

# Menampilkan fitur penting beserta nilainya
print("Fitur yang dipilih dengan Embedded Methods (di atas ambang batas):")
for i in important_features_indices:
    print(f"{x.columns[i]}: {importances[i]}")
# Jika X asli berbentuk DataFrame, maka kita ambil nama kolom

# Mendapatkan nama kolom penting berdasarkan importance
important_features = x_integer.columns[important_features_indices]

# Memindahkan fitur penting ke variabel baru
x_important = x_integer[important_features] # Hanya fitur penting dari data pelatihan

# X_important sekarang berisi hanya fitur penting
print("\nDimensi data pelatihan dengan fitur penting:", x_important.shape)

x_selected = pd.concat([x_important, x['Fitur_12']], axis=1)
x_selected = pd.concat([x_selected, x['Fitur_13']], axis=1)
print(x_selected)

# menggunakan label encoding karena berasumsi bahwa kategori yang ada memiliki urutan yang logis
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
# Melakukan Encoding untuk fitur 12
x_selected['Fitur_12'] = label_encoder.fit_transform(x_selected['Fitur_12'])
# print(label_encoder.inverse_transform(X_Selected[['Fitur_12']]))
# Melakukan Encoding untuk fitur 13
x_selected['Fitur_13'] = label_encoder.fit_transform(x_selected['Fitur_13'])
# print(label_encoder.inverse_transform(X_Selected[['Fitur_13']]))

print(x_selected)

#  Pertama-tama, mari kita mulai dengan menyalin dataset dan menghapus Fitur_12 dan Fitur_13 yang merupakan data hasil encoding.
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Memilih kolom numerik
numeric_columns = x_selected.select_dtypes(include=['float64', 'int64']).columns
numeric_columns =numeric_columns.drop(['Fitur_12', 'Fitur_13'])

# Membuat salinan data untuk menjaga data asli tetap utuh
x_cleaned = x_important.copy()

# mencari nilai batas bawah dan batas atas sehingga Anda dapat menghapus nilai yang berada di luar jangkauan.
# Melihat outlier dengan IQR (Interquartile Range)
for col in numeric_columns:
    Q1 = x_important[col].quantile(0.25)
    Q3 = x_important[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identifikasi outlier
    outliers = x_cleaned[(x_cleaned[col] < lower_bound) | (x_cleaned[col]) > upper_bound]

    # Menghapus outlier dari DataFrame
    x_cleaned = x_cleaned.drop(outliers.index)

    # lihat jumlah data setelah outlier dihilangkan menggunakan kode berikut.
    print(x_cleaned)

# menggunakan teknik SMOTE
# Inisialisasi SMOTE
smote = SMOTE(random_state=42)

# 3. Melakukan oversampling pada dataset
x_resampled, y_resampled = smote.fit_resample(x_cleaned, y)

# Menampilkan distribusi kelas setelah SMOTE
print("Distribusi kelas setelah SMOTE:", Counter(y_resampled))

# Mengubah hasil menjadi DataFrame untuk visualisasi atau analisis lebih lanjut
x_resampled = pd.DataFrame(x_resampled)
y_resampled = pd.Series(y_resampled, name='Target')

# membuat visualisasi distribusi data.
# 1. Visualisasi distribusi data sebelum scaling menggunakan histogram
plt.figure(figsize=(10, 6))
for col in x_resampled.columns:
    sns.histplot(x_resampled[col], kde=True, label=col, bins=30, element='step')
plt.title('Distribusi Data Sebelum Scaling (Histogram)')
plt.legend()
plt.show()

# menggunakan standardisasi agar skala data memiliki skala yang sama serta standar deviasi mendekati satu.
# Scaling: Standarisasi fitur numerik menggunakan StandardScaler
scaler = StandardScaler()

# Melakukan scaling pada fitur penting
x_resampled[important_features] = scaler.fit_transform(x_resampled[important_features])

# 1. Visualisasi distribusi data setelah scaling menggunakan histogram
plt.figure(figsize=(10, 6))
for col in x_resampled.columns:
    sns.histplot(x_resampled[col], kde=True, label=col, bins=30, element='step')
plt.title('Distribusi Data Setelah Scaling (Histogram)')
plt.legend()
plt.show()

x_resampled.describe(include='all')



