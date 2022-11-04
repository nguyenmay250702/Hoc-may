#Bài 2:

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


X = np.array([[2, 8], [2, 5],[1, 2],[5, 8],[7, 3],[6, 4],[8, 4],[4,7]])
print("\n- Dữ liệu đầu vào: \n",X)

kmeans = KMeans(n_clusters=2)
kmeans = kmeans.fit(X)

#tính giá trị dự đoán trên dữ liệu đầu vào là X_test
y_predict = kmeans.predict(X)
print("\nGía trị dự đoán:", y_predict)

print("\n- Tọa độ tâm:\n",kmeans.cluster_centers_)
print("\n- Mức độ phù hợp: ",silhouette_score(X,y_predict))



