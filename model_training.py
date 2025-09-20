import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# import model yang akan digunakan
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

from data_splitting import y_train, y_test

# persiapan data
try:
    df_processed = pd.read_csv("dataset/Bawang Merah preprocessed.csv")
except FileNotFoundError:
    print("--- PENTING ---")
    print("File 'dataset/Bawang Merah_preprocessed.csv' tidak ditemukan.")
    print("Silakan jalankan skrip 04_data_preprocessing.py terlebih dahulu untuk membuatnya.")
    exit()

# menerapkan label encoding
df_lencoder = df_processed.copy()
categorical_feactures = df_lencoder.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for col in categorical_feactures:
    df_lencoder[col] = label_encoder.fit_transform(df_lencoder[col])

# memisahkan fitur dan target
x = df_lencoder.drop(columns=['Harga'])
y = df_lencoder['Harga']

# membagi jadi training sama testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

print("data siap untuk pelatihan model")
print(f"ukuran data latih: {x_train.shape}")
print(f"ukuran data uji: {x_test.shape}")
print("\n" + "="*50 + "\n")

# training

# model least angle regression (Lars)
print("--- Melatih Model 1: Least Angle Regression (Lars) ---")
lars = linear_model.Lars(n_nonzero_coefs=1).fit(x_train, y_train)
print("lars telah di latih")
print("\n" + "-"*50 + "\n")

# model linear regression
print("--- Melatih Model 2: Linear Regression ---")
LR = LinearRegression().fix(x_train, y_train)
print("linear regression telah di latih")
print("\n" + "-"*50 + "\n")

# model gradient boosting regressor
GBR = GradientBoostingRegressor(random_state=184)
GBR.fit(x_train, y_train)
print("Gradient Boosting Regressor telah dilatih.")
print("\n" + "="*50 + "\n")

print("Semua model telah berhasil dilatih menggunakan data training.")
