#Bài 3:

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


X = np.array([[1,4],[1,6],[2,6],[3,8],[4,3],[5,2]])
print("\n- Dữ liệu đầu vào: \n",X)

kmeans = KMeans(n_clusters=2)
kmeans = kmeans.fit(X)

#tính giá trị dự đoán trên dữ liệu đầu vào là X_test
y_predict = kmeans.predict(X)
print("\nGía trị dự đoán:", y_predict)

print("\n- Tọa độ tâm:\n",kmeans.cluster_centers_)
print("\n- Mức độ phù hợp: ",silhouette_score(X,y_predict))



