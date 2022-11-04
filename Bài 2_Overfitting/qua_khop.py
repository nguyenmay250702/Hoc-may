import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression   #import hàm LinearRegression(để tính toán) của gói linear_model trong thư viện sklearn 
from sklearn.model_selection import train_test_split   #import hàm train_test_split(để phân tách dữ liệu) từ gói model_selection của thư viện sklearn
import numpy as np

#đọc dữ liệu
data = pd.read_csv('USA_Housing.csv')   

#chia dữ liệu
dt_Train, dt_Test = train_test_split(data,test_size=0.3, shuffle=False)

#chia dữ liệu thành 5 phần(k=5), random_state=None: không làm gì cả
k = 5
kf = KFold(n_splits=k,random_state=None)

#tính lỗi, y thực tế, y dự đoán
def error(y,y_pred):
    l=[]
    for i in range(0,len(y)):   #độ dài y[0:1]y[0]
       l.append(np.abs(np.array(y[i:i+1]) - np.array(y_pred[i:i+1])))
    return np.mean(l)   #trả về giá trị trung bình độ lệch tìm đc của giá trị dự đoán và thực tế


#tìm ra mô hình huấn luyện tốt nhất(sum nhỏ nhất)
max=9999999
i=1
for train_index,test_index in kf.split(dt_Train):   #duyệt từng phần dữ liệu trong 5 phần dữ liệu đc chia của dt_Train
    X_train,X_test = dt_Train.iloc[train_index,:5],dt_Train.iloc[test_index,:5]
    y_train,y_test = dt_Train.iloc[train_index,5],dt_Train.iloc[test_index,5]

    lr = LinearRegression()
    lr.fit(X_train,y_train)         #huấn luyện mô hình đưa ra tham số vào biến lr
    Y_pred_train = lr.predict(X_train)
    Y_pred_test = lr.predict(X_test)

    sum = error(y_train,Y_pred_train) + error(y_test,Y_pred_test)

    print("sum = ",sum)
    print("max = ",max)
    
    if(sum<max):
        max = sum
        last = i
        regr = lr.fit(X_train,y_train)  #lấy ra mô hình tốt nhất
    i = i+1

print(regr)

"""
y_predict = regr.predict(dt_Test.iloc[:,:5])
y = np.array(dt_Test.iloc[:,5])

print("\nThuc te\t\t\tDu doan\t\t\tchenh lech\n")
for i in range(0,len(y)):
    print("%.2f" %y[i], "\t", y_predict[i], "\t", abs(y[i]-y_predict[i]))


"""










