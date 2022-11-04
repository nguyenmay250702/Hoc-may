# from msilib.schema import ComboBox
# from tkinter import Button, Label, Tk
# import numpy as np
# import pandas as pd
# from sklearn import tree

# from sklearn.model_selection import train_test_split,KFold
# from sklearn import metrics

from cProfile import label
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tkinter.ttk import *
from tkinter import *
from tkinter import messagebox
from sklearn.model_selection import train_test_split,KFold

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

#chuẩn hóa dữ liệu trong data1
data = data_encoder(data1)

#chia dữ liệu thành 3 phần(k=3), random_state=None: không làm gì cả
kf = KFold(n_splits=3,random_state=None)

max=0
stt = 1
#chia dữ liệu dt_Train thành 3 phần
#duyệt trên từng mô hình chia được để tìm ra mô hình tốt(tỉ lệ đúng lớn nhất)
for train_index,test_index in kf.split(data):
    X_train,X_test = data[train_index,:5],data[test_index,:5] 
    y_train,y_test = data[train_index,5:6],data[test_index,5:6]

    
    id3 = tree.DecisionTreeClassifier(criterion='entropy')
    id3.fit(X_train,y_train)         
    Y_pred_train = id3.predict(X_train)
    Y_pred_test = id3.predict(X_test)

    #in ra tỉ lệ đúng của tập train và validation của mỗi mô hình
    print("\n\n- train accuracy:", metrics.accuracy_score(y_train,Y_pred_train))
    print("- validation accuracy:", metrics.accuracy_score(y_test,Y_pred_test))
    
    #tổng tỉ lệ dự đoán đúng trên 2 mô hình càng lớn càng tốt 
    sum = metrics.accuracy_score(y_train,Y_pred_train) + metrics.accuracy_score(y_test,Y_pred_test)
    
    #nếu tổng tỉ lệ dự đoán đúng của mô hình hiện tại lớn hơn mô hình trc đó
    if(sum>max):
        stt_model_best = stt    #lưu lại sô thứ tự của mô hình tốt nhất
        max = sum               #lưu lại tổng tỉ lệ dự đoán cao nhất của mô hình hiện tại
        modelmax_id3 = id3.fit(X_train,y_train)  #huấn luyện mô hình tốt nhất hiện tìm được        
        data_train_best = np.concatenate((X_train, y_train),axis=1) #lưu lại tập train của mô hình tốt nhất
        data_test_best = np.concatenate((X_test, y_test),axis=1) #lưu lại tập test của mô hình tốt nhất
    stt = stt+1

print("\n- Mô hình tốt nhất là mô hình thứ: ", stt_model_best, "\n- Tỉ lệ đúng của mô hình là: ",max)
print("\nTập train:\n", data_train_best)
print("\nTập test:\n", data_test_best)



# def PredictWithID3():
# 	try:
# 		# newData = data_encoder(np.array([[textbox_StudentId.get(), Combobox_Gender.get(), combobox_Pointprs.get(), combobox_Testsc.get(), combobox_Interest.get(), combobox_Attitude()]]))
# 		# newData_Decreased = bestPcaID3.transform(newData)
# 		# Result = modelMaxID3.predict(newData_Decreased)
# 		# lbPredictId3.configure(text=f"{Result[0]}")
# 	except:
# 		messagebox.showinfo("Cảnh báo", "Vui lòng chọn thông tin để dự đoán")


#Tạo form
FORM = Tk()

#Tắt chức năng thay đổi kích thước của form
FORM.resizable(False, False)

#Đặt kích thước cho form
FORM.geometry('470x400')

#Đặt tên cho form
FORM.title("Dự đoán kết quả thi môn học máy của sinh viên")

#Các đối tượng được dùng trong form: Label, Combobox, Button, LabelFrame (Group) 

lbSpace = Label(FORM, text="Thông tin học sinh:", font=("Arial", 10)).grid(row=0, column=0, pady=5, sticky="e")
label_StudentId = Label(FORM, text = "Mã sinh viên ") #mã sinh viên
label_StudentId.grid(row = 1, column = 0, pady = 5)
textbox_StudentId = Entry(FORM)
textbox_StudentId.grid(row = 1, column = 1)

label_Gender = Label(FORM, text = "Giới tính") #Giới tính
label_Gender.grid(row = 2, column = 0, pady = 5)
Combobox_Gender = Combobox (FORM, state="readonly", values= ('Male', 'Female'))
Combobox_Gender.grid(row=2, column=1)

label_Pointprs = Label(FORM, text="Điểm quá trình:").grid(row=3, column=0, pady=5) #Điểm quá trình
combobox_Pointprs = Combobox(FORM, state="readonly", values=('5','6','7','8','9','10'))
combobox_Pointprs.grid(row=3, column=1, pady=5)

label_Testsc = Label(FORM, text="Điểm cuối kì:").grid(row=4, column=0, pady=5) #Điểm thi
combobox_Testsc = Combobox(FORM, state="readonly", values=('4','5','6','7','8'))
combobox_Testsc.grid(row=4, column=1, pady=5)

label_Interest = Label(FORM, text="Mức độ hứng thú với môn học:").grid(row=5, column=0, pady=5) #Độ hứng thú
combobox_Interest = Combobox(FORM, state="readonly", values=('low','medium','high'))
combobox_Interest.grid(row=5, column=1, pady=5)

label_Attitude = Label(FORM, text="Thái độ học:").grid(row=6, column=0, pady=5) #Độ tập trung
combobox_Attitude = Combobox(FORM, state="readonly", values=('focus','lazy'))
combobox_Attitude.grid(row=6, column=1, pady=5)

# btnPredictId3 = Button(FORM, text="Dự đoán kết quả", bg="#C7CBD1", command=PredictWithID3).grid(row=7, column=0, pady=5)
# lbID3 = Label(FORM, text="Kết quả thi:\n(Pass / Faile)").grid(row=7, column=1)
# lbPredictId3 = Label(FORM, text="---------")
# lbPredictId3.grid(row=8, column=1, pady=5)

# lb_id3 = Label(FORM, text="Tỉ lệ dự đoán đúng của ID3: "+'\n'
# 						   +"Accuracy_score: "+str(accuracy_score(Y_test, bestPredID3)*100)+"%"+'\n'
#                            +"Precision: "+str(precision_score(Y_test, bestPredID3, average='macro')*100)+"%"+'\n'
#                            +"Recall: "+str(recall_score(Y_test, bestPredID3, average='macro')*100)+"%"+'\n'
#                            +"F1-score: "+str(f1_score(Y_test, bestPredID3, average='macro')*100)+"%").grid(row=8, column=0, pady=5)

#Gọi vòng lặp sự kiện chính để các hành động có thể diễn ra trên màn hình máy tính

FORM.mainloop()



