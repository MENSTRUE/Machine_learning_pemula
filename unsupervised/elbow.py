from scipy.cluster.vq import kmeans
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer # mengimport kelbowvisualizer untuk visualisasi method elbow
from sklearn.datasets import make_blobs

# membuat dataset buatan
x, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=(1, 10))
visualizer.fit(x)
visualizer.show()