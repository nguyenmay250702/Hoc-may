import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn import metrics
#from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


df = pd.read_csv('weather.csv')     #kiểu dữ liệu DataFrame: cấu trúc dữ liệu 2 chiều
data_X = np.array(df[['outlook', 'temperature', 'humidity', 'wind']].values) #trả về mảng 2 chiều[n,5]

#hàm chuẩn hóa dữ liệu
def data_encoder(data):
    for i, j in enumerate(data):
        for k in range(0, 4):
            if (j[k] == "sunny"):
                j[k] = 0
            elif (j[k] == "overcast"):
                j[k] = 1
            elif (j[k] == "rainy"):
                j[k] = 2
            elif (j[k] == "hot"):
                j[k] = 3
            elif (j[k] == "mild"):
                j[k] = 4
            elif (j[k] == "cool"):
                j[k] = 5
            elif (j[k] == "high"):
                j[k] = 6
            elif (j[k] == "normal"):
                j[k] = 7
            elif (j[k] == "weak"):
                j[k] = 8
            elif (j[k] == "strong"):
                j[k] = 9
    return data

#chuẩn hóa dữ liệu trong data1
X = data_encoder(data_X)

kmeans = KMeans(n_clusters=2)
kmeans = kmeans.fit(X)

#tính giá trị dự đoán trên dữ liệu đầu vào là X_test
y_predict = kmeans.predict(X)
#print(y_predict)

print("- Tọa độ tâm:\n",kmeans.cluster_centers_)
print("- Mức độ phù hợp: ",silhouette_score(X,y_predict))



