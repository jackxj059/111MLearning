from sklearn import cluster, datasets
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
# 讀入鳶尾花資料
iris = datasets.load_iris()
iris_X = iris.data
# KMeans 演算法
kmeans_fit = cluster.KMeans(n_clusters = 3).fit(iris_X)

# 印出分群結果
cluster_labels = kmeans_fit.labels_
print("分群結果：")
print(cluster_labels)
print("---")

# 印出品種看看
iris_y = iris.target
print("真實品種：")
print(iris_y)


plt.rcParams['font.size'] = 14
plt.figure(figsize=(16, 8))
# 以不同顏色畫出原始的 10 群資料
plt.subplot(121)
plt.title(' iris_X Original data')

plt.scatter(iris_X.T[0], iris_X.T[1], c=[1]*150, cmap=plt.cm.Set1)
# 根據重新分成的 5 組來畫出資料
plt.subplot(122)
plt.title('KMeans=3 groups')
plt.scatter(iris_X.T[0], iris_X.T[1], c=iris_y, cmap=plt.cm.Set1)
plt.tight_layout()
plt.show()