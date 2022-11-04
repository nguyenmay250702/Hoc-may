import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression   #import hàm LinearRegression(để tính toán) của gói linear_model trong thư viện sklearn 
from sklearn.model_selection import train_test_split   #import hàm train_test_split(để phân tách dữ liệu) từ gói model_selection của thư viện sklearn
import numpy as np

#đọc vào 1 file
data = pd.read_csv('USA_Housing.csv')   

# data : dữ liệu bị phân tác là
# test_size=0.3 : số xác định kích thước của tập kiểm tra 
# dữ liệu kiểm tra: dt_Test = 30% của tổng số dữ liệu ban đầu "data"
# shuffle=False: dữ liệu đc lấy theo thứ tự từ trên xuống không xáo trộn(shuffle = True: xáo trộn r mới lấy)
# dữ liệu để học: dt_Train = phần còn lại
dt_Train, dt_Test = train_test_split(data,test_size=0.3, shuffle=False)

X_train = dt_Train.iloc[:,:5]   #lấy tất cả các dòng và các cột từ 0->4 {cột: 0,1,2,3,4}
y_train = dt_Train.iloc[:,5]    #lấy tất cả các dòng và cột số 5
X_test = dt_Test.iloc[:,:5]
y_test = dt_Test.iloc[:,5]

"""
print("\nDữ liệu Train: \n", dt_Train)
print("\n\nDữ liệu Test: \n", dt_Test)
print("\n\nCác đặc trưng Train: \n", X_train)
print("\n\nCác nhãn Train: \n", y_train)
"""

#tiến hành học dữ liệu bằng phương thức fix
reg = LinearRegression().fit(X_train,y_train)# thực hiện tính toán trả về model(trả về 1 đối tượng có đầy đủ kết quả)

#reg.score(X_test,y_test)    #hệ số xác định của dự đoán

print("w = ",reg.coef_)     # trả về vecter trọng số w
print("w0 = ",reg.intercept_) #số hạng độc lập (b)

y_pred = reg.predict(X_test)    #giá trị dự đoán với các mẫu dữ liệu X_test
y = np.array(y_test)
#print("\nHệ số xác định: %.2f" % r2_score(y_test,y_pred))
print("\nThuc te\t\t\tDu doan\t\t\tchenh lech\n")
for i in range(0,len(y)):
    print("%.2f" %y[i], "\t", y_pred[i], "\t", abs(y[i]-y_pred[i]))



