# Import library yang diperlukan
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris

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

# 3. Menentukan Ruang Pencarian Hyperparameter (Distribusi)
# Ini adalah "kamus" berisi rentang nilai yang ingin kita coba
# Persis seperti contoh di materi Anda
param_dist = {
    'C': np.logspace(-2, 2, 10),      # Rentang nilai untuk C (10 pilihan)
    'gamma': np.logspace(-4, 1, 10),  # Rentang nilai untuk gamma (10 pilihan)
    'kernel': ['rbf']                # Kernel tetap 'rbf' (1 pilihan)
}
# Total kemungkinan kombinasi = 10 * 10 * 1 = 100 kombinasi

# 4. Inisialisasi dan Menjalankan Random Search
# n_iter=10 berarti kita hanya akan mencoba 10 kombinasi acak dari 100 kemungkinan
# cv=5 berarti kita menggunakan 5-fold cross-validation
# n_jobs=-1 berarti menggunakan semua core prosesor agar lebih cepat
# random_state=42 agar hasil acaknya bisa direproduksi
random_search = RandomizedSearchCV(estimator=model,
                                   param_distributions=param_dist,
                                   n_iter=10,
                                   cv=5,
                                   n_jobs=-1,
                                   verbose=2,
                                   random_state=42)

# Melatih model dengan 10 kombinasi acak pada data latih
random_search.fit(X_train, y_train)

# 5. Menampilkan Hasil Terbaik
print("\n=============================================")
print("Hyperparameter terbaik yang ditemukan dari 10 percobaan acak:")
print(random_search.best_params_)
print("\nSkor cross-validation terbaik (akurasi):")
print(random_search.best_score_)
print("=============================================\n")

# Opsional: Menggunakan model terbaik untuk prediksi di data uji
best_model = random_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
print(f"Akurasi model terbaik pada data uji: {accuracy}")