import pandas as pd
from sklearn.linear_model import LinearRegression   #import hàm LinearRegression(để tính toán) 
from sklearn.model_selection import train_test_split,KFold   #import hàm train_test_split(để phân tách dữ liệu)
from sklearn.metrics import mean_squared_error,r2_score  #import hàm mean_squared_error(để tính trung bình sự chênh lệch giữa gt thực tế và dự đoán)
import numpy as np

#đọc dữ liệu
data = pd.read_csv('winequality_white.csv')   

#chia dữ liệu
dt_Train, dt_Test = train_test_split(data,test_size=0.3, shuffle=False)   

#chia dữ liệu thành 5 phần(k=5), random_state=None: không làm gì cả
kf = KFold(n_splits=5,random_state=None)

max=9999999
#chia dữ liệu dt_Train thành 5 phần
#duyệt trên từng mô hình chia được để tìm ra mô hình tốt(sự sai lệch nhỏ nhất)
for train_index,test_index in kf.split(dt_Train):
    X_train,X_test = dt_Train.iloc[train_index,:11],dt_Train.iloc[test_index,:11] 
    y_train,y_test = dt_Train.iloc[train_index,11],dt_Train.iloc[test_index,11]

    lr = LinearRegression()
    lr.fit(X_train,y_train)         
    Y_pred_train = lr.predict(X_train)
    Y_pred_test = lr.predict(X_test)

    #tổng sai số của mô hình trên 2 tập dữ liệu tập train và tập test
    sum = mean_squared_error(y_train,Y_pred_train) + mean_squared_error(y_test,Y_pred_test) 

    #nếu tổng sai số của mô hình hiện tại nhỏ hơn mô hình trc đó
    if(sum<max):    
        max = sum   
        regr = lr.fit(X_train,y_train)  #huấn luyện mô hình tốt nhất hiện tìm được

#thực hiện tính giá trị dự đoán trên dữ liệu dt_Test(chiếm 30% data đã chia từ đầu)
y_predict = regr.predict(dt_Test.iloc[:,:11])

#giá trị thực tế của dữ liệu dt_Test(chiếm 30% data đã chia từ đầu)
y = np.array(dt_Test.iloc[:,11])










