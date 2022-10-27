from sklearn import cluster, datasets
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# 讀入鳶尾花資料


iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

X_train, X_test, y_train, y_test = train_test_split(iris_X,iris_y,test_size=0.30,random_state=101)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)


#V------------------------ Show plt ------------------- V
plt.rcParams['font.size'] = 14
plt.figure(figsize=(16, 8))
# 以不同顏色畫出原始的 10 群資料
plt.subplot(121)
plt.title(' iris_X Original data')

plt.scatter(X_test.T[0], X_test.T[1], c=[1]*len(X_test), cmap=plt.cm.Set1)
# 根據重新分成的 5 組來畫出資料
plt.subplot(122)
plt.title('knn ')
plt.scatter(X_test.T[0], X_test.T[1], c=pred, cmap=plt.cm.Set1)
plt.tight_layout()
plt.show()