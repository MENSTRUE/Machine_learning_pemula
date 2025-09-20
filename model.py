import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle

# import model yang akan digunakan
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
LR = LinearRegression().fit(x_train, y_train)
print("linear regression telah di latih")
print("\n" + "-"*50 + "\n")

# model gradient boosting regressor
GBR = GradientBoostingRegressor(random_state=184)
GBR.fit(x_train, y_train)
print("Gradient Boosting Regressor telah dilatih.")
print("\n" + "="*50 + "\n")

print("Semua model telah berhasil dilatih menggunakan data training.")

# evaluasi model
print("memuali evaluasi model")

# evaluasi (Lars)
pred_lars = lars.predict(x_test)
mae_lars = mean_absolute_error(y_test, pred_lars)
mse_lars = mean_squared_error(y_test, pred_lars)
r2_lars = r2_score(y_test, pred_lars)

# membuat dictionary
data = {
    'MAE': [mae_lars],
    'MSE': [mse_lars],
    "R2": [r2_lars]
}

# konversi distionary menjadi dataframe
df_results = pd.DataFrame(data, index=['Lars'])
print("\n hasil evaluasi lars")
print(df_results)

# evaluasi (LR)
pred_LR = LR.predict(x_test)
mae_LR = mean_absolute_error(y_test, pred_LR)
mse_LR = mean_squared_error(y_test, pred_LR)
r2_LR = r2_score(y_test, pred_LR)

df_results.loc['Linear Regression'] = [mae_LR, mse_LR, r2_LR]
print("\nPerbandingan Hasil Evaluasi Setelah Menambah Linear Regression:")
print(df_results)

# evaluasi (GBR)
pred_GBR = GBR.predict(x_test)
mae_GBR = mean_absolute_error(y_test, pred_GBR)
mse_GBR = mean_squared_error(y_test, pred_GBR)
r2_GBR = r2_score(y_test, pred_GBR)

# menambahkan hasil evaluasi ke dataframe
df_results.loc['GradientBoostingRegressor'] = [mae_GBR, mse_GBR, r2_GBR]
print("hasil evaluasi terakhir")
print(df_results)
print("\n" + "="*50 + "\n")
print("Proses evaluasi model selesai.")

# menyimpan model
print("Menyimpan Model Terbaik (Gradient Boosting Regressor)")

# menggunakan joblib
joblib.dump(GBR, 'gbr_model.joblib')
print("model berhasil disimpan (gbr_model.joblib)")

# menggunkaan pickle
with open('gbr_model.pkl', 'wb')as file:
    pickle.dump(GBR, file)
print("model berhasil disimpan (pickle)")