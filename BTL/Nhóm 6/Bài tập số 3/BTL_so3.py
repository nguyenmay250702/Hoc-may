from tkinter import *
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


#tính tỉ lệ dự đoán đúng
def tyledung(y_test,y_pred):
    d = 0
    for i in range(len(y_pred)):
        if(y_pred[i]==y_test[i]):
            d = d+1
    return (d/len(y_pred))


df = pd.read_csv('milknew.csv')
X = np.array(df[['pH','Nhiet do','huong vi','mui','chat beo','do trong cua sua','mau sua']].values)    
y = np.array(df['chat luong'])

max_id3 = 0     #lưu giá trị tỉ lệ đúng lớn nhất trong các mô hình khi sd ID3
max_svc = 0
for j in range(1,8):
    pca = PCA(n_components=j)
    pca.fit(X)

    X_bar = pca.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_bar, y, test_size=0.3 , shuffle = False)

    #id3
    id3 = DecisionTreeClassifier(criterion='entropy')
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
form = Tk()             #tạo ra cửa sổ gán vào biến form
form.title("Dự đoán chất lượng sữa:") #thay đổi tiêu đề cửa sổ
form.geometry("760x600")   #kích thước cửa sổ


lable_ten = Label(form, text = "Nhập thông tin cho sản phẩm sữa:", font=("Arial Bold", 10), fg="red")
lable_ten.grid(row = 1, column = 1, pady = 10)

lable_pH = Label(form, text = " pH́:")
lable_pH.grid(row = 2, column = 1, pady = 10)
textbox_pH = Entry(form)
textbox_pH.grid(row = 2, column = 2)

lable_NhietDo = Label(form, text = "Nhiệt độ:")
lable_NhietDo.grid(row = 3, column = 1, pady = 10)
textbox_NhietDo = Entry(form)
textbox_NhietDo.grid(row = 3, column = 2)

lable_HuongVi = Label(form, text = "Hương vị:")
lable_HuongVi.grid(row = 4, column = 1,pady = 10)
textbox_HuongVi = Entry(form)
textbox_HuongVi.grid(row = 4, column = 2)

lable_Mui = Label(form, text = "Mùi:")
lable_Mui.grid(row = 5, column = 1, pady = 10)
textbox_Mui = Entry(form)
textbox_Mui.grid(row = 5, column = 2)

lable_ChatBeo = Label(form, text = "Chất béo trong sữa:")
lable_ChatBeo.grid(row = 6, column = 1, pady = 10 )
textbox_ChatBeo = Entry(form)
textbox_ChatBeo.grid(row = 6, column = 2)

lable_DoTrong = Label(form, text = "Độ trong của sữa:")
lable_DoTrong.grid(row = 7, column = 1, pady = 10 )
textbox_DoTrong = Entry(form)
textbox_DoTrong.grid(row = 7, column = 2)

lable_MauSua = Label(form, text = "Màu của sữa:")
lable_MauSua.grid(row = 8, column = 1, pady = 10 )
textbox_MauSua = Entry(form)
textbox_MauSua.grid(row = 8, column = 2)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ID3

#dudoanid3
lb_id3 = Label(form)
lb_id3.grid(column=1, row=9)
lb_id3.configure(text="\n\nTỉ lệ dự đoán đúng của ID3: "+'\n'
                           +"Precision: "+str(precision_score(y_test, y_pred_best_id3, average='macro')*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_test, y_pred_best_id3, average='macro')*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_test, y_pred_best_id3, average='macro')*100)+"%"+'\n'
                           +"Accuracy: "+str(max_id3*100)+"%")
                        

#hàm dự đoán giá trị theo ID3
def dudoanID3():
    pH = textbox_pH.get()
    NhietDo = textbox_NhietDo.get()
    HuongVi = textbox_HuongVi.get()
    Mui = textbox_Mui.get()
    ChatBeo = textbox_ChatBeo.get()
    DoTrong = textbox_DoTrong.get()
    MauSua = textbox_MauSua.get()
    if((pH == '') or (NhietDo == '') or (HuongVi == '') or (Mui == '') or (ChatBeo == '') or (DoTrong == '') or (MauSua == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = np.array([pH,NhietDo,HuongVi,Mui,ChatBeo,DoTrong, MauSua]).reshape(1, -1)
        X_dudoan_bar = pca_best_id3.transform(X_dudoan)
        y_kqua = modelmax_id3.predict(X_dudoan_bar)
        lb_pred_id3.configure(text= y_kqua)

button_1 = Button(form, text = 'Kết quả dự đoán theo ID3', command = dudoanID3)
button_1.grid(row = 10, column = 1, pady = 20)
lb_pred_id3 = Label(form, text="...")
lb_pred_id3.grid(column=2, row=10)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~SVM

#dudoansvm
lb_svc = Label(form)
lb_svc.grid(column=3, row=9)
lb_svc.configure(text="\n\nTỉ lệ dự đoán đúng của SVM: "+'\n'
                           +"Precision: "+str(precision_score(y_test,  y_pred_best_svc, average='macro')*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_test,  y_pred_best_svc, average='macro')*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_test,  y_pred_best_svc, average='macro')*100)+"%"+'\n'
                           +"Accuracy: "+str(max_svc*100)+"%")

#hàm dự đoán giá trị theo SVM
def dudoanSVM():
    pH = textbox_pH.get()
    NhietDo = textbox_NhietDo.get()
    HuongVi = textbox_HuongVi.get()
    Mui = textbox_Mui.get()
    ChatBeo = textbox_ChatBeo.get()
    DoTrong = textbox_DoTrong.get()
    MauSua = textbox_MauSua.get()
    if((pH == '') or (NhietDo == '') or (HuongVi == '') or (Mui == '') or (ChatBeo == '') or (DoTrong == '') or (MauSua == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = np.array([pH,NhietDo,HuongVi,Mui,ChatBeo,DoTrong, MauSua]).reshape(1, -1)
        X_dudoan_bar = pca_best_svc.transform(X_dudoan)
        y_kqua = modelmax_svc.predict(X_dudoan_bar)
        lb_pred_svm.configure(text= y_kqua)

button_3 = Button(form, text = 'Kết quả dự đoán theo SVM', command = dudoanSVM)
button_3.grid(row = 10, column = 3, pady = 20)
lb_pred_svm = Label(form, text="...")
lb_pred_svm.grid(column=4, row=10)


form.mainloop() #hiển thị cửa sổ

