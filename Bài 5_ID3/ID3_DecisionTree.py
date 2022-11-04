import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv('weather.csv')     #kiểu dữ liệu DataFrame: cấu trúc dữ liệu 2 chiều
data1 = np.array(df[['outlook', 'temperature', 'humidity', 'wind', 'play']].values) #trả về mảng 2 chiều[n,5]

#hàm chuẩn hóa dữ liệu
def data_encoder(data):
    for i, j in enumerate(data):
        for k in range(0, 5):
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
            """elif (j[k] == "no"):
                j[k] = 11
            elif (j[k] == "yes"):
                j[k] = 12"""
    return data

#chuẩn hóa dữ liệu trong data1
data = data_encoder(data1)

#chia dữ liệu thành 2 phần: train 70%, test = 30%
dt_Train, dt_Test = train_test_split(data, test_size=0.3 , shuffle = False)

X_train = dt_Train[:, :4]
y_train = dt_Train[:, 4]
X_test = dt_Test[:, :4]
y_test = dt_Test[:, 4]

#max_depth=3: chiều sâu đối đa của cây
#min_samples_leaf=5: số lượng mẫu tối thiểu cần thiết có ở nút lá
#clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
#clf = tree.DecisionTreeClassifier(criterion = "gini")

#khai báo cây quyết định
#clf = tree.DecisionTreeClassifier()
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
