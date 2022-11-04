from tkinter import *
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


df = pd.read_csv('milknew.csv')
X = np.array(df[['pH','Nhiet do','huong vi','mui','chat beo','do trong cua sua','mau sua']].values)    
y = np.array(df['chat luong'])

#duyệt tìm ra mô hình tốt nhất
max_id3 = 0    
for j in range(1,8):
    pca = PCA(n_components=j)
    pca.fit(X)

    X_bar = pca.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_bar, y, test_size=0.3 , shuffle = False)

    #id3
    id3 = DecisionTreeClassifier(criterion='entropy')
    id3.fit(X_train, y_train)
    y_pred_id3 = id3.predict(X_test)
    rate_id3 = metrics.accuracy_score(y_test, y_pred_id3)       #tính tỉ lệ dự đoán đúng

    if(rate_id3 >max_id3):
        num_pca_id3 = j     #lưu lại số thuộc tính tốt nhất
        pca_best_id3 = pca  #lưu lại mô hình pca tốt nhất
        max_id3 = rate_id3  #lưu lại tỉ lệ dự đoán đúng của mô hình tốt nhất
        modelmax_id3 = id3  #mô hình có tỉ lệ đúng lớn nhất
        y_pred_best_id3 = y_pred_id3
        

#form
form = Tk()          
form.title("Dự đoán chất lượng sữa:") 
form.geometry("600x450")

lable_NhietDo = Label(form, text = "Dự đoán chất lượng sữa", font=("Comic Sans MS Bold", 19), fg = "red").grid(row = 0, column = 1, pady = 10, sticky="e")

group1 = LabelFrame(form, text="Nhập thông tin cho sản phẩm sữa")
group1.grid(row=1, column=1, padx=50, pady=30)
group2 = LabelFrame(form, bd=0)
group2.grid(row=1, column=2)
group3 = LabelFrame(group2, text="Đánh giá mô hình được chọn:")
group3.grid(row=1, column=1, pady=20)

lable_pH = Label(group1, text = " pH́:").grid(row = 1, column = 1, pady = 10,sticky="e")
textbox_pH = Entry(group1)
textbox_pH.grid(row = 1, column = 2, padx = 20)

lable_NhietDo = Label(group1, text = "Nhiệt độ:").grid(row = 2, column = 1, pady = 10,sticky="e")
textbox_NhietDo = Entry(group1)
textbox_NhietDo.grid(row = 2, column = 2)

lable_HuongVi = Label(group1, text = "Hương vị:").grid(row = 3, column = 1,pady = 10,sticky="e")
textbox_HuongVi = Entry(group1)
textbox_HuongVi.grid(row = 3, column = 2)

lable_Mui = Label(group1, text = "Mùi:").grid(row = 4, column = 1, pady = 10,sticky="e")
textbox_Mui = Entry(group1)
textbox_Mui.grid(row = 4, column = 2)

lable_ChatBeo = Label(group1, text = "Chất béo trong sữa:").grid(row = 5, column = 1, pady = 10,sticky="e")
textbox_ChatBeo = Entry(group1)
textbox_ChatBeo.grid(row = 5, column = 2)

lable_DoTrong = Label(group1, text = "Độ trong của sữa:").grid(row = 6, column = 1, pady = 10,sticky="e")
textbox_DoTrong = Entry(group1)
textbox_DoTrong.grid(row = 6, column = 2)

lable_MauSua = Label(group1, text = "Màu của sữa:").grid(row = 7, column = 1, pady = 10,sticky="e")
textbox_MauSua = Entry(group1)
textbox_MauSua.grid(row = 7, column = 2)

lable_MauSua = Label(group2, text = "Chất lượng sữa (high/low)", font=("Arial italic", 8)).grid(row = 3, column = 1, pady = 10)


#Đánh giá độ đo
lb_id3 = Label(group3)
lb_id3.grid(row=0, column=1, padx = 35, pady = 20)
lb_id3.configure(text=  "Precision: "+str(metrics.precision_score(y_test, y_pred_best_id3, average='macro')*100)+"%"+'\n'
                        +"\nRecall: "+str(metrics.recall_score(y_test, y_pred_best_id3, average='macro')*100)+"%"+'\n'
                        +"\nF1-score: "+str(metrics.f1_score(y_test, y_pred_best_id3, average='macro')*100)+"%"+'\n'
                        +"\nAccuracy: "+str(metrics.accuracy_score(y_test, y_pred_best_id3)*100)+"%")


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
        lb_pred_id3.configure(text= y_kqua[0])

button_1 = Button(group2, text = 'Kết quả dự đoán', font=("Arial Bold", 9), fg = "GreenYellow", bg = "black", command = dudoanID3)
button_1.grid(row = 2, column = 1)
lb_pred_id3 = Label(group2, text="...", font=("Arial Bold", 9), fg = "white", bg = "SlateGray4")
lb_pred_id3.grid(row=4, column=1)

form.mainloop()

