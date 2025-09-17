import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv("dataset/Bawang Merah.csv")
train.head()
print(train.head())

# data cleaning

# mengidentifikasi informasi dataset
# Menampilkan ringkasan informasi dari dataset
print(train.info())

# analis deskriptif
# Menampilkan statistik deskriptif dari dataset
desc = train.describe(include="all")
print(desc)

# memeriksa jumlah nilai yang hilang di setiap kolom (missing value)
missing_values = train.isnull().sum()
nilai_hilang = missing_values[missing_values > 0]
print(nilai_hilang)

# Menmgatasi missing value

# Pisahkan kolom yang missing value < 75% dan >= 75%
# less = missing_values[missing_values < 822].index
# over = missing_values[missing_values >= 822].index
# jika tidak ingin ribet begini mending
threshold = 0.75 * len(train)
less = missing_values[missing_values < threshold].index
over = missing_values[missing_values >= threshold].index

# mengisi nilai hilang dengan median untuk kolom numerik
# 1. memilih nama nama kolom dari Dataframe train yang memiliki tipe data numeric dari subset kolom yang ditentukan oleh less
numeric_feactures = train[less].select_dtypes(include=['number']).columns

# 2. hitung nilai median untuk kolom (column) dan disimpan di variable
median_values = train[numeric_feactures].median()
print("median yang akan digunakan untuk mengisi data hilang :")
print(median_values)

# 3. baris ini yaitu kita mengisi semua nilai yang hilang (NaN) pada kolom kolom numerik tersebut dengan nilai median dari masing masing kolom
# train[numeris_feactures] = train[numeris_feactures].fillna(train[numeris_feactures].median())
train[numeric_feactures] = train[numeric_feactures].fillna(median_values)

# ada cara lain menangani yang serupa pada data bertipe object atau string (tdak sama dengan ketika kita menangani tipe data numeric)
# ada 2 cara
# 1. mengisi missing value dengan modus (nilai yang paling sering muncul)
kategorial_feactures = train[less].select_dtypes(include=['object']).columns

for column in kategorial_feactures:
    train[column] = train[column].fillna(train[column].mode()[0])

# menghapus kolom tersebut sesuai dengan nama fitur yang sudah ditentukan sebelumnya / terlalu banyak nilai yang hilang
df = train.drop(columns=over)

# 2. mengisi dengan kategori baru (misalnya "unknown" atau "missing")

# lakukan pemeriksaan terhadap data yang sudah melewati tahapan verifikasi missing value
missing_values = df.isnull().sum()
missing_values[missing_values > 0]

print(missing_values)

# mengatasi outliers

# periksa terlebih dahulu apakah dataset yang digunakan memiliki outlier atau tidak


for feature in numeric_feactures:
    plt.figure(figsize=(10,6))
    sns.boxplot(x=df[feature])
    plt.title(f'Box plot of {feature}')
    plt.show()

# pada chart nilai yang berada dibawah batas bawah atau diatas batas atas dianggap sebagai outlier
# anda memiliki 2 pilihan
# 1. anda dapat memilih dan menghapus outlier
# 2. menggantinya dengan nilai yang lebih moderat (seperti batas terdekat), atau menerapkan tranformasi

# adapun langkah umum untuk mendeteksi dan menangani outlier menggunakan metode IQR
# 1. menghitung IQR,Q1 dan Q3
# a. Q1(Quartile 1) nilai di persentil 25 data
# b. Q3(Quartile 3): nilai persentil ke 75 data
# IQR = rentang antara Q3 dan Q1(IQR = Q3-Q1)

# 2. menentukan batas bawah dan batas atas
# a. batas bawah : Q1 - 1.5*IQR
# b. Batas atas : Q3 + 1.5*IQR

# mengidentifikasi outliers menggunakan IQR
Q1 = df[numeric_feactures].quantile(0.25)
Q3 = df[numeric_feactures].quantile(0.75)
IQR = Q3 - Q1

# filter dataframe untuk hanya menyimpan baris yang tidak mengandung outliers pada kolom numeric
condition = ~((df[numeric_feactures] < (Q1 - 1.5 * IQR)) | (df[numeric_feactures] > (Q3 + 1.5 * IQR))).any(axis=1)
df_filtered_numeric = df.loc[condition, numeric_feactures]

# menggabungkan kembali dengan kolom kategorial
categorical_feactures = df.select_dtypes(include=['object']).columns
df = pd.concat([df_filtered_numeric, df.loc[condition, categorical_feactures]], axis=1)

for feature in numeric_feactures:
    plt.figure(figsize=(10,6))
    sns.boxplot(x=df[feature])
    plt.title(f'Box plot of {feature} - AFTER')
    plt.show()
