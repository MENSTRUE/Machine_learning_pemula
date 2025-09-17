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

# # mengisi nilai hilang dengan median untuk kolom numerik
# # memilih nama nama kolom dari Dataframe train yang memiliki tipe data numeric dari subset kolom yang ditentukan oleh less
numeris_feactures = train[less].select_dtypes(include=['number']).columns
# # baris ini yaitu kita mengisi semua nilai yang hilang (NaN) pada kolom kolom numerik tersebut dengan nilai median dari masing masing kolom
train[numeris_feactures] = train[numeris_feactures].fillna(train[numeris_feactures].median())


