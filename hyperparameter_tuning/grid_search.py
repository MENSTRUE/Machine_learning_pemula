# Import library yang diperlukan
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
import pandas as pd

# 1. Memuat Dataset
# Kita akan menggunakan dataset Iris yang simpel
iris = load_iris()
X = iris.data
y = iris.target

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Menentukan Model
# Kita akan menggunakan model Support Vector Classifier (SVC)
model = SVC()

# 3. Menentukan Ruang Hyperparameter (Grid)
# Ini adalah "kamus" berisi semua nilai yang ingin kita coba
param_grid = {
    'C': [0.1, 1, 10, 100],  # Parameter regularisasi
    'gamma': [1, 0.1, 0.01, 0.001], # Koefisien kernel
    'kernel': ['rbf', 'linear'] # Jenis kernel yang akan diuji
}

# Total kombinasi yang akan dicoba adalah: 4 (nilai C) * 4 (nilai gamma) * 2 (nilai kernel) = 32 kombinasi

# 4. Inisialisasi dan Menjalankan Grid Search
# cv=5 berarti kita menggunakan 5-fold cross-validation
# n_jobs=-1 berarti menggunakan semua core prosesor agar lebih cepat
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Melatih model dengan semua kombinasi hyperparameter pada data latih
grid_search.fit(X_train, y_train)

# 5. Menampilkan Hasil Terbaik
print("\n=============================================")
print("Hyperparameter terbaik yang ditemukan:")
print(grid_search.best_params_)
print("\nSkor cross-validation terbaik (akurasi):")
print(grid_search.best_score_)
print("=============================================\n")

# Opsional: Menggunakan model terbaik untuk prediksi di data uji
best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
print(f"Akurasi model terbaik pada data uji: {accuracy}")