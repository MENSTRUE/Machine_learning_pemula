# =============================================================================
# Import Library yang Diperlukan
# =============================================================================
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV

# =============================================================================
# 1. Memuat dan Mempersiapkan Data
# =============================================================================
print("--- 1. Memuat dan Mempersiapkan Data Klasifikasi ---")
X, y = fetch_openml(name='credit-g', version=1, return_X_y=True, as_frame=True)

le = LabelEncoder()
y = le.fit_transform(y)

X_encoded = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

print(f"Shape of training data: {X_train.shape}")
print(f"Shape of testing data: {X_test.shape}\n")

# =============================================================================
# 2. Model Awal (Tanpa Tuning)
# =============================================================================
print("--- 2. Melatih Model Awal (Foundation Model) ---")
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

initial_score = rf_classifier.score(X_test, y_test)
print(f"Initial accuracy on test set (without tuning): {initial_score:.4f}\n")

# =============================================================================
# 3. Hyperparameter Tuning dengan Grid Search
# =============================================================================
print("--- 3. Menjalankan Grid Search ---")
start_time_grid = time.time()
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_rf_grid = grid_search.best_estimator_
grid_search_score = best_rf_grid.score(X_test, y_test)
end_time_grid = time.time()

print(f"\nBest parameters (Grid Search): {grid_search.best_params_}")
print(f"Accuracy after Grid Search: {grid_search_score:.4f}")
print(f"Waktu eksekusi Grid Search: {end_time_grid - start_time_grid:.2f} detik\n")

# =============================================================================
# 4. Hyperparameter Tuning dengan Random Search
# =============================================================================
print("--- 4. Menjalankan Random Search ---")
start_time_random = time.time()
param_dist = {
    'n_estimators': np.linspace(100, 500, 5, dtype=int),
    'max_depth': np.linspace(10, 50, 5, dtype=int),
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
random_search = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_dist, n_iter=20, cv=3, n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train, y_train)

best_rf_random = random_search.best_estimator_
random_search_score = best_rf_random.score(X_test, y_test)
end_time_random = time.time()

print(f"\nBest parameters (Random Search): {random_search.best_params_}")
print(f"Accuracy after Random Search: {random_search_score:.4f}")
print(f"Waktu eksekusi Random Search: {end_time_random - start_time_random:.2f} detik\n")

# =============================================================================
# 5. Hyperparameter Tuning dengan Bayesian Optimization
# =============================================================================
print("--- 5. Menjalankan Bayesian Optimization ---")
# Pastikan Anda sudah menginstal scikit-optimize: pip install scikit-optimize
start_time_bayes = time.time()
param_space = {
    'n_estimators': (100, 500),
    'max_depth': (10, 50),
    'min_samples_split': (2, 10),
    'criterion': ['gini', 'entropy']
}
bayes_search = BayesSearchCV(estimator=rf_classifier, search_spaces=param_space, n_iter=32, cv=3, n_jobs=-1, verbose=2, random_state=42)
bayes_search.fit(X_train, y_train)

best_rf_bayes = bayes_search.best_estimator_
bayes_search_score = best_rf_bayes.score(X_test, y_test)
end_time_bayes = time.time()

print(f"\nBest parameters (Bayesian Optimization): {bayes_search.best_params_}")
print(f"Accuracy after Bayesian Optimization: {bayes_search_score:.4f}")
print(f"Waktu eksekusi Bayesian Optimization: {end_time_bayes - start_time_bayes:.2f} detik\n")