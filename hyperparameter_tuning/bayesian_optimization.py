# Import library yang diperlukan
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_iris

# 1. Memuat Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Menentukan Model
model = SVC()

# 3. Menentukan Ruang Pencarian Hyperparameter (Search Space)
# Kita mendefinisikan rentang nilai, bukan daftar nilai tetap.
# Ini memberikan fleksibilitas yang jauh lebih besar.
search_spaces = {
    'C': Real(1e-6, 1e+6, prior='log-uniform'),
    'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    'degree': Integer(1, 8),
    'kernel': Categorical(['linear', 'poly', 'rbf']),
}

# 4. Inisialisasi dan Menjalankan Bayesian Optimization (BayesSearchCV)
# n_iter=30 berarti kita memberikan "budget" 30 kali percobaan cerdas.
# cv=5 berarti kita menggunakan 5-fold cross-validation.
bayes_search = BayesSearchCV(estimator=model,
                             search_spaces=search_spaces,
                             n_iter=30,
                             cv=5,
                             n_jobs=-1,
                             verbose=2,
                             random_state=42)

# Menjalankan pencarian cerdas pada data latih
bayes_search.fit(X_train, y_train)

# 5. Menampilkan Hasil Terbaik
print("\n=============================================")
print("Hyperparameter terbaik yang ditemukan melalui Bayesian Optimization:")
print(bayes_search.best_params_)
print("\nSkor cross-validation terbaik (akurasi):")
print(bayes_search.best_score_)
print("=============================================\n")

# Opsional: Menggunakan model terbaik untuk prediksi di data uji
best_model = bayes_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
print(f"Akurasi model terbaik pada data uji: {accuracy}")