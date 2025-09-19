import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# memuat data
print("memuat data")
df_awal = pd.read_csv("dataset/Bawang Merah.csv")
print("Data awal berhasil dimuat")
print(f"Ukuran data awal: {df_awal.shape}")

# penanganan