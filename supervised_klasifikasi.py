import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from django.db.models.expressions import result
from pandas.core.interchange.from_dataframe import categorical_column_to_series
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from EDA_ExDA import correlation_matrix, label_encoder
from data_splitting import x_test, y_train, y_test
from normalisasi_standarisasi import scaler

# 2. memuat data

# gantilah id file dengan id dari google drive url
file_id = '19IfOP0QmCHccMu8A6B2fCUpFqZwCxuzO'

# buat url unduhan langsung
download_url = f'https://drive.google.com/uc?id={file_id}'

# baca file csv dari url
data = pd.read_csv(download_url)

# tampilan data frame
data.head()

# tampilkan informasi umum
print("\n Informasi Dataset:")
data.info()

# cek missing value
print("\nMissing values per fitur:")
print(data.isnull().sum())

# hapus kolom 'rownumber', 'customerid', dan 'surname'
data = data.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# tampilkan dataframe untuk memastikan kolom telah dihapus
print(data.head())

# 3. exploratory data analysis (EDA)
# distribusi fitur numeric

num_feactures = data.select_dtypes(include=[np.number])
plt.figure(figsize=(14, 10))
for i, column in enumerate(num_feactures.columns, 1):
    plt.subplot(3, 4, i)
    sns.histplot(data[column], bins=30, kde=True, color='blue')
    plt.title(f'Distribusi {column}')
plt.tight_layout()
plt.show()

# distribusi fitur kategorikal
cat_features = data.select_dtypes(include=['object'])
plt.figure(figsize=(14, 8))
for i, column in enumerate(cat_features.columns, 1):
    plt.subplot(2, 4, i)
    sns.countplot(y=data[column], palette='viridis')
    plt.title(f'Distribusi {column}')
plt.tight_layout()
plt.show()

# heatmap korelasi untuk fitur numerik
plt.figure(figsize=(12, 10))
correlation_matrix = num_feactures.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Heatmap korelasi')
plt.show()


# pairplot untuk fitur numeric
sns.pairplot(num_feactures)
plt.show()

# visualisasi distribusi variable target
plt.figure(figsize=(8, 4))
sns.countplot(x='Exited', data=data, palette='viridis')
plt.title('Distribusi variable target (Exited)')
plt.show()

# 4. label encoder
# buat instance labelencoder
label_encoder = LabelEncoder()

# list kolom kategorikal yang perlu di encode
categorical_columns = ['Geography', 'Gender']

# encode kolom kategorical
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# tampilkan data frame untuk memastikan encoding telah diterapkan
print(data.head())

# 5. Data splitting
# buat instance MinMaxScaler
scaler = MinMaxScaler()

# normalisasi semua kolom numerik
numeric_columns = data.select_dtypes(include=['int64', 'Float64']).columns
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# pisahkan fitur(x) dan target(y)
x = data.drop(columns=['Exited'])
y = data['Exited']

# split data menjadi set pelatihan dan uji
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# tampilkan bentuk set pelatihan dan set uji untuk memastikan split
print(f"Training set shape: x_train={x_train.shape}, y_train={y_train.shape}")
print(f"Test set shape: x_test{x_test.shape}, y_test{y_test.shape}")

# 6. pelatihan model
# definisikan setiap klasifikasi secara terpisah
knn = KNeighborsClassifier().fit(x_train, y_train)
dt = DecisionTreeClassifier().fit(x_train, y_train)
rf = RandomForestClassifier().fit(x_train, y_train)
svm = SVC().fit(x_train, y_train)
nb = GaussianNB().fit(x_train, y_train)

print("model training selesai")


# 7. evaluasi model
# fungsi untuk mengevaluasi dan mengembalikan hasil sebagai kamus
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    results = {
        'confussion matrix': cm,
        'True positive (TP)': tp,
        'False positif (FP)': fp,
        'False negative (FN)': fn,
        'True negative (TN)': tn,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precission': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred)
    }
    return results

# mengevaluasi setiap model dan mengumpulkan hasilnya
results = {
    'K-nearest neighbors (KNN)': evaluate_model(knn, x_test, y_test),
    'Decission tree (DT)': evaluate_model(dt, x_test, y_test),
    'Random forest (RF)': evaluate_model(rf, x_test, y_test),
    'Support vector machine (SVM)': evaluate_model(svm, x_test, y_test),
    'Naive Bayes (NB)': evaluate_model(nb, x_test, y_test)
}

# buat data frame untuk meringkas hasil
summary_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precission', 'Recall', 'F1-Score'])

# isi dataframe dengan hasil
rows = []
for model_name, metrics in results.items():
    rows.append({
        'Model': model_name,
        'Accuracy': metrics['Accuracy'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'F1-Score': metrics['F1-Score']
    })

# konversi daftar kamus ke dataframe
summary_df = pd.DataFrame(rows)

# tampilkan dataframe
print(summary_df)