import pandas as pd

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
numeris_feactures = train[less].select_dtypes(include=['number']).columns

# 2. hitung nilai median untuk kolom (column) dan disimpan di variable
median_values = train[numeris_feactures].median()
print("median yang akan digunakan untuk mengisi data hilang :")
print(median_values)

# 3. baris ini yaitu kita mengisi semua nilai yang hilang (NaN) pada kolom kolom numerik tersebut dengan nilai median dari masing masing kolom
# train[numeris_feactures] = train[numeris_feactures].fillna(train[numeris_feactures].median())
train[numeris_feactures] = train[numeris_feactures].fillna(median_values)

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


