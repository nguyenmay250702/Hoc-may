#from future import print_function 
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('weather.csv')

# X_data = np.array(df[['outlook', 'temperature', 'humidity', 'wind', 'play']].values)
le = LabelEncoder()
outlook = le.fit_transform(df['outlook'].values)
temperature = le.fit_transform(df['temperature'].values)
humidity = le.fit_transform(df['humidity'].values)
wind = le.fit_transform(df['wind'].values)
play = df['play'].values

X_data = np.array([outlook, temperature, humidity, wind, play])
print(X_data)

data = X_data.T
X = data[:, :4]
y = data[:, 4]

tree = DecisionTreeClassifier()
tree.fit(X, y)
y_pred = tree.predict(X)

print("Thực tế \t Dự đoán")
for i in range (0, len(y)):
    print("%.5s" % y[i], "\t\t", y_pred[i])
count = 0
for i in range(0,len(y_pred)):
    if(y[i] == y_pred[i]):
        count = count + 1

print('Ty le du doan dung:', count/len(y_pred)*100, "%")
