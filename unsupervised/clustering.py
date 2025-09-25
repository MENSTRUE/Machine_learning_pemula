# import liberary yang dibutuhkan

import pandas as pd # Mengimpor pustaka pandas untuk manipulasi dan analisis data
import matplotlib.pyplot as plt # Mengimpor pustaka matplotlib untuk visualisasi grafik
from yellowbrick.cluster import KElbowVisualizer # Mengimpor KElbowVisualizer untuk visualisasi metode Elbow

from sklearn.cluster import KMeans, DBSCAN # Mengimpor algoritma KMeans dan DBSCAN untuk clustering
from sklearn.metrics import silhouette_score # Mengimpor silhouette_score untuk mengevaluasi hasil clustering

# data loading
# membaca dataset pelanggan mall dari url dan menampilkan 5 baris pertama
df = pd.read_csv('https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/main/ML%20Pemula/Mall_Customers.csv')
print(df.head())

# menampilakn informasi tentang dataset, termasuk jumlah baris, kolom, tipe data, dan jumlah nilai non-null
print(df.info())

# menampilkan statistik deskriptif dari dataset untuk kolom numerik
print(df.describe())


# exploratory data analysis
# menghitung distribusi gender dan menampilkan pie chart untuk visualisasi
plt.figure(figsize=(7, 7))
plt.pie(df['Gender'].value_counts(), labels=['Female', 'Male'], autopct='%1.1f%%', startangle=90)
plt.title('Gender distribution')
plt.show()

# bining / mengubah data angka yang detail jadi beberapa kelompok kategori
# mengelompokan data usia pelanggan ke dalam kategori dan menghitung jumlah pelanggan di setiap kategori
age18_25 = df.Age[(df.Age >= 18) & (df.Age <= 25)]
age26_35 = df.Age[(df.Age >= 26) & (df.Age <= 35)]
age36_45 = df.Age[(df.Age >= 36) & (df.Age <= 45)]
age46_55 = df.Age[(df.Age >= 46) & (df.Age <= 55)]
age55above = df.Age[(df.Age >= 56)]

# menyusun data untuk plotting
x = ["18-25", "26-35", "36-45", "46-55", "55+"]
y = [len(age18_25.values), len(age26_35.values), len(age36_45.values), len(age46_55.values), len(age55above.values)]

# membuat bar chart untuk distribusi usia pelanggan
plt.figure(figsize=(15, 6))
plt.bar(x, y, color=['red', 'green', 'blue', 'cyan', 'yellow'])
plt.title("customer and their ages")
plt.xlabel("Age")
plt.ylabel("Number of customers")

# menambahkan label jumlah pelanggan diatas setiap bar
for i in range(len(x)):
    plt.text(i, y[i], y[i], ha="center", va="bottom")

plt.show()

# mengelompokan pendapatan tahunan pelanggan ke dalam kategori dan menghitung jumlah pelanggan di setiap kategori
ai0_30 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 0) & (df["Annual Income (k$)"] <= 30)]
ai31_60 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 31) & (df["Annual Income (k$)"] <= 60)]
ai61_90 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 61) &(df["Annual Income (k$)"] <= 90)]
ai91_120 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 91) &(df["Annual Income (k$)"] <= 120)]
ai121_150 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 121) &(df["Annual Income (k$)"] <= 150)]

# menyusun data untuk plotting
aix = ["$ 0 - 30,000", "$ 30,001 - 60,000", "$ 60,001 - 90,000", "$ 90,001 - 120,000", "$ 120,001 - 150,000"]
aiy = [len(ai0_30.values), len(ai31_60.values), len(ai61_90.values), len(ai91_120.values), len(ai121_150.values)]

# membuat bar chart untuk distribusi pendapatan tahunan pelanggan
plt.figure(figsize=(15, 6))
plt.bar(aix, aiy, color=['red', 'green', 'blue', 'cyan', 'yellow'])
plt.title("customer and their annual income")
plt.xlabel("Annual Income")
plt.ylabel("Numbers of customers")
plt.xticks(rotation=45) # memutar label sumbu x agar lebih mudah dibaca

# menambahkan label jumlah pelanggan diatas setiap bar
for i in range(len(aix)):
    plt.text(i, aiy[i], aiy[i], ha='center', va='bottom')

plt.show()

# Data splitting
# mengambil kolom 'annual income (k$)' dan 'spending score (1-100)' dari dataset dan menyimpannya dalam array X
x = df.iloc[:, [3, 4]].values

# Menampilkan data yang diambil dalam format DataFrame dengan nama kolom yang sesuai
print(pd.DataFrame(x, columns=['Annual Income (k$)', 'Spending Score (1-100)']))

# menghindari overfitting dengan memilih jumlah cluster yang sesuai dengan struktur data.
# Inisialisasi model KMeans tanpa parameter awal

# elbow method

kmeans = KMeans()

# Inisialisasi visualizer KElbow untuk menentukan jumlah cluster optimal
visualizer = KElbowVisualizer(kmeans, k=(1, 10))

# Fit visualizer dengan data untuk menemukan jumlah cluster optimal
visualizer.fit(x)

# Menampilkan grafik elbow untuk analisis
visualizer.show()

# cluster modeling (k-means clustering)
from sklearn.cluster import KMeans


# Inisialisasi dan melatih model KMeans dengan jumlah cluster = 4
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(x)

# Mendapatkan label cluster
labels = kmeans.labels_

# Mendapatkan jumlah cluster
k = 4

# Fungsi untuk analisis karakteristik cluster
def analyze_clusters(x, labels, k):
    print("Analisis karakteristik setiap cluster:")
    for cluster_id in range(k):
        # Mengambil data untuk cluster saat ini
        cluster_data = x[labels == cluster_id]

        # Menghitung rata-rata untuk setiap fitur dalam cluster
        mean_income = cluster_data[:, 0].mean() # Rata-rata Annual Income
        mean_spending = cluster_data[:, 1].mean() # Rata-rata Spending Score

        print(f"\nCluster {cluster_id + 1}:")
        print(f"Rata-rata Annual Income (k$): {mean_income:.2f}")
        print(f"Rata-rata Spending Score (1-100): {mean_spending:.2f}")

# analisis karakteristik setiap cluster
analyze_clusters(x, labels, k)

# menunjukkan pendapatan tahunan dan skor belanja rata-rata yang mewakili pusat dari masing-masing cluster.

# menentukan posisi centroid
centroids = kmeans.cluster_centers_

# visualisasi cluster
plt.figure(figsize=(12, 8))

# plot data
plt.scatter(x[:, 0], x[:, 1], c=labels, cmap="viridis", s=50, alpha=0.6, edgecolors='w', marker='o')

# Plot centroid
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='x', label='centroids')

# menambahkan label centroid pada plot
for i, centroid in enumerate(centroids):
    plt.text(centroid[0], centroid[1], f'centroid {i+1}', color='red', fontsize=12, ha='center', va='center')

 # Menambahkan judul dan label
plt.title('Visualisasi Cluster dengan Centroid')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

plt.show()

# Menampilkan nilai centroid
print("Nilai Centroids:")
for i, centroid in enumerate(centroids):
    print(f"centroid {i+1}: Annual income = {centroid[0]:.2f}, spending score = {centroid[1]:.2f}")