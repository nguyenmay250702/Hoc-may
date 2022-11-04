import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


df = pd.read_csv('winequality_white.csv')     #kiểu dữ liệu DataFrame: cấu trúc dữ liệu 2 chiều
X = np.array(df[['do axit co dinh','do axit de bay hoi','axit citric','duong du','clorua','luu huynh dioxit tu do','tong luu huynh dioxit','ty trong','pH','sunfat','ruou']].values)

kmeans = KMeans(n_clusters=2)
kmeans = kmeans.fit(X)

#tính giá trị dự đoán trên dữ liệu đầu vào là X_test
y_predict = kmeans.predict(X)

print("- Tọa độ tâm: \n",kmeans.cluster_centers_)
print("\n- Mức độ phù hợp: ",silhouette_score(X,y_predict))
