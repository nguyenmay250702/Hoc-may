import pandas as pd
from sklearn.linear_model import LinearRegression   
from sklearn.model_selection import train_test_split
#from sklearn import preprocessing

import numpy as np

#đọc vào 1 file
dt = pd.read_csv('cars.csv')
data1 = np.array(dt[['mua','bao tri','cua','nguoi','lug_boot','su an toan','kha nang chap nhan']].values)

"""
le = preprocessing.LabelEncoder()
le.fit(["vhigh","high", "med", "low", "2", "3", "4", "5more","more","small", "big"])
LabelEncoder()
#le.transform(np.array.(X_train))

print(le.transform(np.array(X_train)))
"""

"""
#tiến hành học dữ liệu bằng phương thức fix
reg = LinearRegression().fit(X_train,y_train)

#giá trị dự đoán với tập dữ liệu X_test
y_pred = reg.predict(X_test)    
y = np.array(y_test)

print("\nThuc te\t\t\tDu doan\t\t\tchenh lech\n")
for i in range(0,len(y)):   #lặp từ 0->len(y)-1
    print("%.2f" %y[i], "\t\t", y_pred[i], "\t\t", abs(y[i]-y_pred[i]))

"""
