import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv('Final_result.csv')     
data1 = np.array(df[['gender', 'point_process', 'test_score', 'interest', 'attitude', 'result']].values) #lấy ra các cột có tên đc nêu

#hàm chuẩn hóa dữ liệu
def data_encoder(data):
    for i, j in enumerate(data):
        for k in range(0, 5):
            if (j[k] == "Female"):
                j[k] = 0
            elif (j[k] == "Male"):
                j[k] = 1
            elif (j[k] == "low"):
                j[k] = 2
            elif (j[k] == "medium"):
                j[k] = 3
            elif (j[k] == "high"):
                j[k] = 4
            elif (j[k] == "focus"):
                j[k] = 5
            elif (j[k] == "lazy"):
                j[k] = 6               
    return data

#chuẩn hóa dữ liệu trong data1
data = data_encoder(data1)

#chia dữ liệu thành 2 phần: train 70%, test = 30%
dt_Train, dt_Test = train_test_split(data, test_size=0.3 , shuffle = False)

X_train = dt_Train[:, :5]
y_train = dt_Train[:, 5]
X_test = dt_Test[:, :5]
y_test = dt_Test[:, 5]

clf = tree.DecisionTreeClassifier(criterion = "entropy")

#sd cây vừa tìm đc để huấn luyện trên tập train
clf = clf.fit(X_train, y_train)

#tính giá trị dự đoán trên dữ liệu đầu vào là X_test
y_predict = clf.predict(X_test)

print("Độ chính xác precision: ",metrics.precision_score(y_test, y_predict, average='macro'))  #độ chính xác
print("Độ thu hồi recall: ",metrics.recall_score(y_test, y_predict, average='micro')) #độ thu hồi
print("Gía trị trung bình giữa độ chính xác và thu hồi: ",metrics.f1_score(y_test, y_predict, average='weighted'))  #giá trị trung bình giữa độ chính xác và độ thu hồi

count = 0
print("Thực tế\t\tDự đoán")
for i in range(0,len(y_predict)):
    print(y_test[i],"\t\t", y_predict[i])
    if(y_test[i] == y_predict[i]):
        count = count +1

print('Ty le du doan dung: ', count/len(y_predict))
