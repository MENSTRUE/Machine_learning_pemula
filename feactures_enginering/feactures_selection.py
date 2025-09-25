import numpy as np
import pandas as pd
from numpy.ma.core import indices
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine


# memuat dataset wine quality
data = load_wine()
x, y = data.data, data.target

# mengubah menjadi dataframe untuk analisis yang lebih mudah
df = pd.DataFrame(x, columns=data.feature_names)
df['target'] = y
print(df)

# pembagian data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ------------------------------------------- filter methods -------------------------------------------
# menggunakan selectkbest
filter_selector = SelectKBest(score_func=chi2, k=2) #memilih 2 fitur terbaik
x_train_filter = filter_selector.fit_transform(x_train, y_train)
x_test_filter = filter_selector.transform(x_test)

print("Fitur yang dipilih dengan Filter Methods:", filter_selector.get_support(indices=True))

# menggunakan RFE (Recursive feature elimination)
model = LogisticRegression(solver='lbfgs', max_iter=5000)
rfe_selector = RFE(model, n_features_to_select=2) #memilih 2 fitur
x_train_rfe = rfe_selector.fit_transform(x_train, y_train)
x_test_rfe = rfe_selector.transform(x_test)

print("Fitur yang dipilih dengan Wrapper Methods:", rfe_selector.get_support(indices=True))

# ----------------------------------------- embedded methods -----------------------------------------
# menggunakan random forest untuk mendapatkan fitur penting
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# mendapatkan fitur penting
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Menentukan ambang batas untuk fitur penting
threshold = 0.05 # misalnya diambang batas 5%
important_features_indices = [i for i in range(len(importances)) if importances[i] >= threshold]

# Memindahkan fitur penting ke variabel baru
x_important = x_train[:, important_features_indices] # Hanya fitur penting dari data pelatihan
x_test_important = x_test[:, important_features_indices] # Hanya fitur penting dari data pengujian

# Mencetak fitur yang dipilih
print("Fitur yang dipilih dengan Embedded Methods (di atas ambang batas):")
for i in important_features_indices:
    print(f"{data.feature_names[i]}: {importances[i]}")

# X_important sekarang berisi hanya fitur penting
print("\nDimensi data pelatihan dengan fitur penting:", x_important.shape)
print("Dimensi data pengujian dengan fitur penting:", x_test_important.shape)

# membuat sebuah fungsi agar dapat digunakan berulang kali.
# Evaluasi dengan fitur terpilih dari masing-masing metode

def evaluate_model(x_train, x_test, y_train, y_test, model):
    model.fit(x_train, y_train)
    accuracy = model.score (x_test, y_test)
    return accuracy

# melatih model machine learning berdasarkan fitur yang telah ditentukan oleh masing-masing metode feature selection
# Model Logistic Regression untuk Filter Methods
logistic_model_filter = LogisticRegression(max_iter=200)
accuracy_filter = evaluate_model(x_train_filter, x_test_filter, y_train, y_test, logistic_model_filter)

# Model Logistic Regression untuk Wrapper Methods
logistic_model_rfe = LogisticRegression(max_iter=200)
accuracy_rfe = evaluate_model(x_train_rfe, x_test_rfe, y_train, y_test, logistic_model_rfe)

# Model Random Forest untuk Embedded Methods
accuracy_rf = evaluate_model(x_important, x_test_important, y_train, y_test, rf_model)

print(f"\nAkurasi Model dengan Filter Methods: {accuracy_filter:.2f}")
print(f"Akurasi Model dengan Wrapper Methods: {accuracy_rfe:.2f}")
print(f"Akurasi Model dengan Embedded Methods: {accuracy_rf:.2f}")
