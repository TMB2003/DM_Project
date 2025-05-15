from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import numpy as np

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Agglomerative Clustering Model
model = AgglomerativeClustering(n_clusters=3)
labels = model.fit_predict(X)

# Match cluster labels with true labels
mapped_labels = np.zeros_like(labels)
for i in range(3):
    mask = (labels == i)
    mapped_labels[mask] = mode(y[mask], keepdims=True)[0]

print("Agglomerative Clustering Accuracy:", accuracy_score(y, mapped_labels))
