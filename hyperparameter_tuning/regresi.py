# =============================================================================
# Import Library yang Diperlukan
# =============================================================================
import time
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV

# =============================================================================
# 1. Memuat dan Mempersiapkan Data
# =============================================================================
print("--- 1. Memuat dan Mempersiapkan Data Regresi ---")
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Shape of training data: {X_train.shape}")
print(f"Shape of testing data: {X_test.shape}\n")

# =============================================================================
# 2. Model Awal (Tanpa Tuning)
# =============================================================================
print("--- 2. Melatih Model Awal (Foundation Model) ---")
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train, y_train)

y_pred_initial = rf_regressor.predict(X_test)
initial_mse = mean_squared_error(y_test, y_pred_initial)
print(f"Initial MSE on test set (without tuning): {initial_mse:.4f}\n")

# =============================================================================
# 3. Hyperparameter Tuning dengan Grid Search
# =============================================================================
print("--- 3. Menjalankan Grid Search ---")
start_time_grid = time.time()
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_rf_grid = grid_search.best_estimator_
y_pred_grid = best_rf_grid.predict(X_test)
grid_search_mse = mean_squared_error(y_test, y_pred_grid)
end_time_grid = time.time()

print(f"\nBest parameters (Grid Search): {grid_search.best_params_}")
print(f"MSE after Grid Search: {grid_search_mse:.4f}")
print(f"Waktu eksekusi Grid Search: {end_time_grid - start_time_grid:.2f} detik\n")

# =============================================================================
# 4. Hyperparameter Tuning dengan Random Search
# =============================================================================
print("--- 4. Menjalankan Random Search ---")
start_time_random = time.time()
param_dist = {
    'n_estimators': np.arange(100, 501, 100),
    'max_depth': [None] + list(np.arange(10, 51, 10)),
    'min_samples_split': np.arange(2, 11, 2),
    'min_samples_leaf': np.arange(1, 5),
    'bootstrap': [True, False]
}
random_search = RandomizedSearchCV(estimator=rf_regressor, param_distributions=param_dist, n_iter=50, cv=3, n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train, y_train)

best_rf_random = random_search.best_estimator_
y_pred_random = best_rf_random.predict(X_test)
random_search_mse = mean_squared_error(y_test, y_pred_random)
end_time_random = time.time()

print(f"\nBest parameters (Random Search): {random_search.best_params_}")
print(f"MSE after Random Search: {random_search_mse:.4f}")
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
    'min_samples_leaf': (1, 4),
    'bootstrap': [True, False]
}
bayes_search = BayesSearchCV(estimator=rf_regressor, search_spaces=param_space, n_iter=32, cv=3, n_jobs=-1, verbose=2, random_state=42)
bayes_search.fit(X_train, y_train)

best_rf_bayes = bayes_search.best_estimator_
y_pred_bayes = best_rf_bayes.predict(X_test)
bayes_mse = mean_squared_error(y_test, y_pred_bayes)
end_time_bayes = time.time()

print(f"\nBest parameters (Bayesian Optimization): {bayes_search.best_params_}")
print(f"MSE after Bayesian Optimization: {bayes_mse:.4f}")
print(f"Waktu eksekusi Bayesian Optimization: {end_time_bayes - start_time_bayes:.2f} detik\n")