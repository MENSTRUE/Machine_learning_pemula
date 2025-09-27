from sklearn.preprocessing import MinMaxScaler, StandardScaler

# contoh data
data = [[10], [2], [30], [40], [50]]

# Min-Max scalling
min_max_scaler = MinMaxScaler()
scaled_min_max = min_max_scaler.fit_transform(data)
print("min max scaling:\n", scaled_min_max)

# standariation
standard_scaler = StandardScaler()
scaled_standard = standard_scaler.fit_transform(data)
print("\nStandarization:\n", scaled_standard)