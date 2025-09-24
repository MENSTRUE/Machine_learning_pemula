import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

from sklearn.preprocessing import StandardScaler

import sklearn
from sklearn import datasets

from sklearn import linear_model

from model import mse_GBR

# data loading

df_train = pd.read_csv("../dataset/flood/flood/train.csv")
print(df_train)

df_test = pd.read_csv("../dataset/flood/flood/test.csv")
print(df_test)

# Data cleaning dan transformation

# menampilkan ringkasan informasi dari dataset
df_train.info()

# menampilkan statistik deskriptif dari dataset
print(df_train.describe(include="all"))

missing_values = df_train.isnull().sum()
print(missing_values[missing_values > 0])

# memeriksa apakah dataset yang digunakan memiliki outlier atau tidak
for feacture in df_train.columns:
    plt.figure(figsize=(10, 6))
    plt.boxplot(x=df_train[feacture])
    plt.title(f'Box plot of {feacture}')
    plt.show()

# memeriksa outlier menggunakan metode IQR
# contoh sederhana untuk mengidentifikasi outliers menggunakan IQR
Q1 = df_train.quantile(0.25)
Q3 = df_train.quantile(0.75)
IQR = Q3 - Q1

# filter dataframe untuk hanya menyimpan baris yang tidak mengandung outliers pada kolom numeric
condition = ~((df_train < (Q1 - 1.5 * IQR)) | (df_train > (Q3 + 1.5 * IQR))).any(axis=1)
df = df_train.loc[condition, df_train.columns]

# memastikan hanya data dengan tipe numerikal yang akan di proses
numeric_features = df.select_dtypes(include=['number']).columns
print(numeric_features)

# standarisasi fitur numerik
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# memastikan data sudah bersih
# mengidentifikasi baris duplicat
duplicates = df.duplicated()

print("baris duplikat:")
print(df[duplicates])

df.describe(imclude='all')

# memvisualisasikan agar bentuk nya terlihat lebih jelas
# menghitung jumlah variable
num_vars = df.shape[1]

# menentukan jumlah baris dan kolom untuk grid subplot
n_cols = 4 #jumlah kolom yang diinginkan
n_rows = -(-num_vars // n_cols) #celling divission untuk menentukan jumlah baris

# membuat subplot
fig, axes = plt.subplot(n_rows, n_cols, figsize=(20, n_rows * 4))

# flatten axes array untuk memudahkan iterasi jika di perlukan
axes = axes.flatten()

# plot setiap variable
for i, column in enumerate(df.drop(columns=["id"]).columns):
    df[column].hist(ax=axes[i], bins=20, edgecolor='black')
    axes[i].set_title(column)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')

# menghapus subplot yang tuidak terpakai (jika ada)
for i, column in enumerate(df.drop(columns=['id']).columns):
    df[column].hist(ax=axes[i], bins=20, edgecolor='black')
    axes[i].set_title(column)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')

# menghapus sublopt yang tidak terpakai (jika ada)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# menyesuaikan layout agar lebih rapi
plt.tight_layout()
plt.show()

# menghitung korelasi antara variable target dan semua variable lainnya
target_corr = df.corr()['floodProbability']

# (opsional) mengurutkan hasil korelasi berdasarkan kekuatan korelasi
target_corr_sorted = target_corr.abs().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
target_corr_sorted.plot(kind='bar')
plt.title(f'corellation with flood probability')
plt.xlabel('variables')
plt.ylabel('correlation coefficient')
plt.show()

# split data
# memisahkan fitur (X) dan target(Y)
x = df.drop(columns=['floodprobability'])
y = df['floodprobability']

from sklearn.model_selection import  train_test_split

# membagi dataset menjadi training dan data testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# menghitung panjang/jumlah data
print("jumlah data", len(x))

# menghitung panjang/jumlah data pada x_train
print("jumlah data latih:", len(x_train))

# menghitung panjang/jumlah data pada x_test
print("jumlah data test:", len(x_test))

# Lars
lars =  linear_model.Lars(n_nonzero_coefs=1).fit(x_train, y_train)

pred_lars = lars.predict(x_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae_lars = mean_absolute_error(y_test, pred_lars)
mse_lars = mean_squared_error(y_test, pred_lars)
r2_lars = r2_score(y_test, pred_lars)

print(f"MAE: {mae_lars}")
print(f"MSE: {mse_lars}")
print(f"R²: {r2_lars}")

# Membuat dictionary untuk menyimpan hasil evaluasi
data = {
    'MAE': [mae_lars],
    'MSE': [mse_lars],
    'R2': [r2_lars]
}

# Konversi dictionary menjadi DataFrame
df_results = pd.DataFrame(data, index=['Lars'])
print(df_results)

# Linear regression
from sklearn.linear_model import LinearRegression
LR = LinearRegression().fit(x_train, y_train)

pred_LR = LR.predict(x_test)

mae_LR = mean_absolute_error(y_test, pred_LR)
mse_LR = mean_squared_error(y_test, pred_LR)
r2_LR = r2_score(y_test, pred_LR)

print(f"MAE: {mae_LR}")
print(f"MSE: {mse_LR}")
print(f"R²: {r2_LR}")

df_results.loc['Linear Regression'] = [mae_LR, mse_LR, r2_LR]
print(df_results)

# gradientboostingregressor
from sklearn.ensemble import GradientBoostingRegressor

GBR = GradientBoostingRegressor(random_state=184)
GBR.fit(x_train, y_train)

pred_GBR = GBR.predict(x_test)

mae_GBR = mean_absolute_error(y_test, pred_GBR)
mse_GBR = mean_squared_error(y_test, pred_GBR)
r2_GBR = r2_score(y_test, pred_GBR)

print(f"MAE: {mae_GBR}")
print(f"MSE: {mse_GBR}")
print(f"R²: {r2_GBR}")

df_results.loc['GradientBoostingRegressor'] = [mae_GBR, mse_GBR, r2_GBR]
print(df_results)

