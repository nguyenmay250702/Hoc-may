import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn.svm import SVC
from sklearn import tree

#tính tỉ lệ dự đoán đúng
def tyledung(y_test,y_pred):
    d = 0
    for i in range(len(y_pred)):
        if(y_pred[i]==y_test[i]):
            d = d+1
    return (d/len(y_pred))


#đọc dữ liệu từ file weather.csv
df = pd.read_csv('milknew.csv')

#lấy ra các mẫu dữ liệu X
data_X = np.array(df[['pH','Nhiet do','huong vi','mui','chat beo','do trong cua sua','mau sua']].values)

#lấy ra nhãn của các mẫu dữ liệu
data_Y = np.array(df['chat luong'])
 
max = 0     #lưu giá trị lớn nhất tỉ lệ dự đoán đúng của các mô hình

#tìm ra thuộc tính quan trọng nhất
for j in range(1,8):
    print("lan",j)
    pca = decomposition.PCA(n_components=j) #sd pca để chọn ra j thuộc tính tốt nhất
    pca.fit(data_X)

    Xbar = pca.transform(data_X) #áp dụng giảm kích thước cho X chỉ còn j thuộc tính
    X_train,X_test,y_train,y_test = train_test_split(Xbar,data_Y,test_size=0.3,shuffle=False)

    #svc = SVC(kernel = 'linear')
    svc = tree.DecisionTreeClassifier(criterion = "entropy")
    
    svc.fit(X_train,y_train)
    y_pred = svc.predict(X_test)
    rate = tyledung(y_test,y_pred)
    print('Tỷ lệ dự đoán đúng: ', rate)

    if(rate >max):
        num_pca = j     #lưu lại số thuộc tính tốt nhất
        pca_best = pca  #lưu lại mô hình pca tốt nhất
        max = rate      #lưu lại tỉ lệ dự đoán đúng của mô hình tốt nhất
        modelmax = svc  #mô hình có tỉ lệ đúng lớn nhất

print("\nTỉ lệ dự đoán đúng của mô hình tốt nhất:", max,"\nSố thuộc tính tốt nhất = ", num_pca)

sample_test = [['6.6','35','1','0','1','0','254']]
sample_pca=pca_best.transform(sample_test)
y_pred = modelmax.predict(sample_pca)
print('Nhãn của sampel_test:', y_pred)
    































