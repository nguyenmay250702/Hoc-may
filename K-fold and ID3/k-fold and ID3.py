import numpy as np
import pandas as pd
from sklearn import tree

from sklearn.model_selection import train_test_split,KFold
from sklearn import metrics

df = pd.read_csv('Final_result.csv')     
data1 = np.array(df[['gender', 'point_process', 'test_score', 'interest', 'attitude', 'result']].values) #lấy ra các cột có tên đc nêu


#hàm chuẩn hóa dữ liệu(đưa dữ liệu dạng chuỗi về số thì máy mới học đc)
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

def data_encoder_y(data):
    for k in range(0, len(data)):
        if (data[k] == "Faile"):
            data[k] = 7
        elif (data[k] == "Pass"):
            data[k] = 8               
    return data

#chuẩn hóa dữ liệu trong data1
data = data_encoder(data1)

#chia dữ liệu sau khi chuẩn hóa thành 2 phần: train 70%, test = 30%
dt_Train, dt_Test = train_test_split(data, test_size=0.3 , shuffle = False)

#chia dữ liệu thành 3 phần(k=3), random_state=None: không làm gì cả
kf = KFold(n_splits=3)

max=9999999
count = 1
#chia dữ liệu dt_Train thành 3 phần
#duyệt trên từng mô hình chia được để tìm ra mô hình tốt(sự sai lệch nhỏ nhất)
for train_index,test_index in kf.split(dt_Train):
    X_train,X_test = dt_Train[train_index,:5],dt_Train[test_index,:5] 
    y_train,y_test = dt_Train[train_index,5],dt_Train[test_index,5]

    #in ra 3 mô hình được duyệt
    """
    print("\n\n- Mô hình thứ: ", count)
    print("- Tập train: \n", np.concatenate((X_train, y_train),axis=1))
    print("- Tập test: \n", np.concatenate((X_test, y_test),axis=1))
    count= count + 1
    """

    id3 = tree.DecisionTreeClassifier(criterion='entropy')
    id3.fit(X_train,y_train)         
    Y_pred_train = id3.predict(X_train)
    Y_pred_test = id3.predict(X_test)

    #tổng sai số của mô hình trên 2 tập dữ liệu tập train và tập test 
    sum = metrics.mean_squared_error(data_encoder_y(y_train),data_encoder_y(Y_pred_train)) + metrics.mean_squared_error(data_encoder_y(y_test),data_encoder_y(Y_pred_test))

    if(sum<max):    
        max = sum   #lưu lại tổng tỉ lệ dự đoán cao nhất của mô hình hiện tại
        modelmax_id3 = id3  #huấn luyện mô hình tốt nhất hiện tìm được


#thực hiện tính giá trị dự đoán trên dữ liệu dt_Test(chiếm 30% data đã chia từ đầu)
y_predict = modelmax_id3.predict(dt_Test[:,:5])

#giá trị thực tế của dữ liệu dt_Test(chiếm 30% data đã chia từ đầu)
y = dt_Test[:,5]

#đánh giá mô hình dự trên các độ đo
print("\nprecision = ",metrics.precision_score(y, y_predict, average='macro'))  #độ chính xác
print("recall = ",metrics.recall_score(y, y_predict, average='micro')) #độ thu hồi
print("f1 = ",metrics.f1_score(y, y_predict, average='weighted'))  #giá trị trung bình giữa độ chính xác và độ thu hồi
print("accuracy = ",metrics.accuracy_score(y, y_predict)) 


