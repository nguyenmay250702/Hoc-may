import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import decomposition
from sklearn.metrics import classification_report

#tính tỉ lệ dự đoán đúng
def tyledung(y_test,y_pred):
    d = 0
    for i in range(len(y_pred)):
        if(y_pred[i]==y_test[i]):
            d = d+1
    return (d/len(y_pred))

#chuẩn hóa dữ liệu
def data_encoder(X):
    for i, j in enumerate(X):
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
    return X

#đọc dữ liệu từ file weather.csv
df = pd.read_csv('weather.csv')

#lấy ra các mẫu dữ liệu X
data_X = np.array(df[['outlook', 'temperature', 'humidity', 'wind']].values)

#lấy ra nhãn của các mẫu dữ liệu
data_Y = np.array(df['play'])

#chuẩn hóa dữ liệu X_data
X = data_encoder(data_X)
 
max = 0     #lưu giá trị lớn nhất tỉ lệ dự đoán đúng của các mô hình

#tìm ra thuộc tính quan trọng nhất
for j in range(1,5):
    print("lan",j)
    pca = decomposition.PCA(n_components=j) #sd pca để chọn ra j thuộc tính tốt nhất
    pca.fit(X)

    Xbar = pca.transform(X) #áp dụng giảm kích thước cho X chỉ còn j thuộc tính
    X_train,X_test,y_train,y_test = train_test_split(Xbar,data_Y,test_size=0.3,shuffle=False)

    svc = SVC(kernel = 'linear')
    svc.fit(X_train,y_train)
    y_pred = svc.predict(X_test)
    rate = tyledung(y_test,y_pred)
    print('Tỷ lệ dự đoán đúng: ', rate)

    if(rate >max):
        num_pca = j     #lưu lại số thuộc tính tốt nhất
        pca_best = pca  #lưu lại mô hình pca tốt nhất
        max = rate      #lưu lại tỉ lệ dự đoán đúng của mô hình tốt nhất
        modelmax = svc  #mô hình có tỉ lệ đúng lớn nhất
        y_pred_best = y_pred 

print("\nTỉ lệ dự đoán đúng của mô hình tốt nhất:", max,"\nSố thuộc tính tốt nhất = ", num_pca)

sample_test = [['sunny','hot','high','weak']]
sample_encoder = data_encoder(sample_test)
sample_pca=pca_best.transform(sample_encoder)
y_pred = modelmax.predict(sample_pca)
print('Nhãn của sampel_test:', y_pred)

print(classification_report(y_test, y_pred_best,zero_division = 1))
    
































