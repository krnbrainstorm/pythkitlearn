import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import datasets
import matplotlib.pyplot as plt


iris = datasets.load_iris()
X = iris.data
y = iris.target


bandwidth = estimate_bandwidth(X, quantile=0.5)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X,y)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

plt.figure(1)
plt.clf()
for k in range(n_clusters_):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], '.')

plt.title('Estimated number of clusters: %d' % n_clusters_)
#changed to 3d display
plt.show()
