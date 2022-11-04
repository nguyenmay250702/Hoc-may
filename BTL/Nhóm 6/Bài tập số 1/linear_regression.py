import pandas as pd
from sklearn.linear_model import LinearRegression   
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np

#đọc vào 1 file
data = pd.read_csv('winequality_white.csv')  

#Chia dữ liệu: dt_Train = 70%data(sd để huấn luyện), dt_Test = 30%data(sd để kiểm tra)
dt_Train, dt_Test = train_test_split(data,test_size=0.3, shuffle=False)

X_train = dt_Train.iloc[:,:11]   
y_train = dt_Train.iloc[:,11]   
X_test = dt_Test.iloc[:,:11]
y_test = dt_Test.iloc[:,11]

#tiến hành học dữ liệu bằng phương thức fix, reg là biến sd để lưu mô hình
reg = LinearRegression().fit(X_train,y_train)

#giá trị dự đoán với tập dữ liệu X_test
y_pred = reg.predict(X_test)    
y = np.array(y_test)

#print("Sai số trung bình: ", mean_squared_error(y_test,y_pred))
#print("w = ",reg.coef_)     # trả về vecter trọng số w
#print("w0 = ",reg.intercept_) #số hạng độc lập (b)
print("\nĐánh giá sự phù hợp của mô hình: %.2f" % r2_score(y_test,y_pred))

print("\nThuc te\t\t\tDu doan\t\t\tchenh lech\n")
for i in range(0,len(y)):   #lặp từ 0->len(y)-1
    print("%.2f" %y[i], "\t\t", y_pred[i], "\t\t", abs(y[i]-y_pred[i]))


