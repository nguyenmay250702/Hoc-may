import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report

from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import svm


data = pd.read_csv('winequality_white.csv')     

#chia dữ liệu thành 2 phần: train 70%, test = 30%
dt_Train, dt_Test = train_test_split(data, test_size=0.3 , shuffle = False)

X_train = dt_Train.iloc[:,:11]   
y_train = dt_Train.iloc[:,11]   
X_test = dt_Test.iloc[:,:11]
y_test = dt_Test.iloc[:,11]

#tìm ra siêu phẳng bằng thuật toán "perceptron"
#clf = Perceptron()

#xây dựng cây quyết định bằng thuật toán ID3 sd hàm "entropy"
#clf = tree.DecisionTreeClassifier(criterion = "entropy")

#xây dựng cây quyết định bằng thuật toán CART sd giá trị "gini"
#clf = tree.DecisionTreeClassifier(criterion = "gini")

#tìm ra siêu phẳng sd thuật toán SVM
clf = svm.SVC(kernel='linear')

clf = clf.fit(X_train, y_train)

#tính giá trị dự đoán trên dữ liệu đầu vào là X_test
y_predict = clf.predict(X_test)

print("Thuật toán svm:")
print("Độ chính xác trung bình: precision = %.2f"%metrics.precision_score(y_test, y_predict, average='macro',zero_division = 1))
print("Độ thu hồi trung bình: recall = %.2f"%metrics.recall_score(y_test, y_predict, average='macro', zero_division = 1))
print("Gía trị trung bình giữa độ chính xác và thu hồi: f1= %.2f"%metrics.f1_score(y_test, y_predict, average='macro', zero_division = 1))  

#print(classification_report(y_test, y_predict))





