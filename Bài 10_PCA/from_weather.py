from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


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

#tính tỉ lệ dự đoán đúng
def tyledung(y_test,y_pred):
    d = 0
    for i in range(len(y_pred)):
        if(y_pred[i]==y_test[i]):
            d = d+1
    return (d/len(y_pred))


df = pd.read_csv('weather.csv')
X = data_encoder(np.array(df[['outlook','temperature','humidity','wind']].values))    
y = np.array(df['play'])

max_id3 = 0
max_svc = 0
for j in range(1,5):
    pca = decomposition.PCA(n_components=j)     #sd pca để chọn ra j thuộc tính tốt nhất
    pca.fit(X)

    X_bar = pca.transform(X)    #áp dụng giảm kích thước cho X chỉ còn j thuộc tính
    X_train, X_test, y_train, y_test = train_test_split(X_bar, y, test_size=0.3 , shuffle = False)

    #id3
    id3 = DecisionTreeClassifier(criterion='entropy')   #xây dựng cây quyết định sd tt ID3
    id3.fit(X_train, y_train)
    y_pred_id3 = id3.predict(X_test)
    rate_id3 = tyledung(y_test,y_pred_id3)

    #svm
    svc = SVC(kernel = 'linear')
    svc.fit(X_train, y_train)
    y_pred_svc = svc.predict(X_test)
    rate_svc = tyledung(y_test,y_pred_svc)

    if(rate_id3 >max_id3):
        num_pca_id3 = j     #lưu lại số thuộc tính tốt nhất
        pca_best_id3 = pca  #lưu lại mô hình pca tốt nhất
        max_id3 = rate_id3      #lưu lại tỉ lệ dự đoán đúng của mô hình tốt nhất
        modelmax_id3 = id3  #mô hình có tỉ lệ đúng lớn nhất
        y_pred_best_id3 = y_pred_id3

    if(rate_svc >max_svc):
        num_pca_svc = j     #lưu lại số thuộc tính tốt nhất
        pca_best_svc = pca  #lưu lại mô hình pca tốt nhất
        max_svc = rate_svc      #lưu lại tỉ lệ dự đoán đúng của mô hình tốt nhất
        modelmax_svc = svc  #mô hình có tỉ lệ đúng lớn nhất
        y_pred_best_svc = y_pred_svc


#form
form = Tk()             #tạo ra csht
form.title("Dự đoán thời tiết:") #thay đổi tiêu đề cửa sổ
form.geometry("700x600")   #kích thước cửa sổ


lable_ten = Label(form, text = "Nhập thông tin thời tiết:", font=("Arial Bold", 10), fg="red")
lable_ten.grid(row = 1, column = 1, pady = 10)

lable_outlook = Label(form, text = "outlook:")
lable_outlook.grid(row = 2, column = 1, pady = 10)
textbox_outlook = Entry(form)
textbox_outlook.grid(row = 2, column = 2)

lable_temperature = Label(form, text = "temperature:")
lable_temperature.grid(row = 3, column = 1, pady = 10)
textbox_temperature = Entry(form)
textbox_temperature.grid(row = 3, column = 2)

lable_humidity = Label(form, text = "humidity:")
lable_humidity.grid(row = 4, column = 1,pady = 10)
textbox_humidity = Entry(form)
textbox_humidity.grid(row = 4, column = 2)

lable_wind = Label(form, text = "wind:")
lable_wind.grid(row = 5, column = 1, pady = 10)
textbox_wind = Entry(form)
textbox_wind.grid(row = 5, column = 2)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ID3

#dudoanid3
lb_id3 = Label(form)
lb_id3.grid(column=1, row=6, pady = 10)
#ghi đè configure
lb_id3.configure(text="Tỉ lệ dự đoán đúng của ID3: "+'\n'
                           +"Precision: "+str(precision_score(y_test, y_pred_best_id3, average='macro', zero_division = 1)*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_test, y_pred_best_id3, average='macro', zero_division = 1)*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_test, y_pred_best_id3, average='macro', zero_division = 1)*100)+"%"+'\n')

#hàm dự đoán giá trị theo ID3
def dudoanID3():
    outlook = textbox_outlook.get()
    temperature = textbox_temperature.get()
    humidity = textbox_humidity.get()
    wind = textbox_wind.get()
    
    if((outlook == '') or (temperature == '') or (humidity == '') or (wind == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = data_encoder(np.array([outlook,temperature,humidity,wind]).reshape(1, -1))
        X_dudoan_bar = pca_best_id3.transform(X_dudoan)
        y_kqua = modelmax_id3.predict(X_dudoan_bar)
        lb_pred_id3.configure(text= y_kqua)

button_1 = Button(form, text = 'Kết quả dự đoán theo ID3', command = dudoanID3)
button_1.grid(row = 7, column = 1, pady = 20)
lb_pred_id3 = Label(form, text="...")
lb_pred_id3.grid(column=2, row=7)

#hàm tính tỉ lệ dự đoán đúng theo ID3
def khanangID3():
    lb_rate_id3.configure(text= max_id3)

button_2 = Button(form, text = 'Khả năng dự đoán đúng ', command = khanangID3)
button_2.grid(row = 8, column = 1, padx = 30)
lb_rate_id3 = Label(form, text="...")
lb_rate_id3.grid(column=2, row=8)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~SVM

#dudoansvm
lb_svc = Label(form)
lb_svc.grid(column=3, row=6, pady = 10)
lb_svc.configure(text="Tỉ lệ dự đoán đúng của SVM: "+'\n'
                           +"Precision: "+str(precision_score(y_test,  y_pred_best_svc, average='macro',zero_division = 1)*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_test,  y_pred_best_svc, average='macro',zero_division = 1)*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_test,  y_pred_best_svc, average='macro',zero_division = 1)*100)+"%"+'\n')

#hàm dự đoán giá trị theo SVM
def dudoanSVM():
    outlook = textbox_outlook.get()
    temperature = textbox_temperature.get()
    humidity = textbox_humidity.get()
    wind = textbox_wind.get()
    
    if((outlook == '') or (temperature == '') or (humidity == '') or (wind == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = data_encoder(np.array([outlook,temperature,humidity,wind]).reshape(1, -1))
        X_dudoan_bar = pca_best_id3.transform(X_dudoan)
        y_kqua = modelmax_id3.predict(X_dudoan_bar)
        lb_pred_svm.configure(text= y_kqua)

button_3 = Button(form, text = 'Kết quả dự đoán theo SVM', command = dudoanSVM)
button_3.grid(row = 7, column = 3, pady = 20)
lb_pred_svm = Label(form, text="...")
lb_pred_svm.grid(column=4, row=7)

#hàm tính tỉ lệ dự đoán đúng theo SVM
def khanangSVM():
    lb_rate_svm.configure(text= max_svc)

button_4 = Button(form, text = 'Khả năng dự đoán đúng ', command = khanangSVM)
button_4.grid(row = 8, column = 3, padx = 30)
lb_rate_svm = Label(form, text="...")
lb_rate_svm.grid(column=4, row=8)


form.mainloop() #hiển thị cửa sổ

