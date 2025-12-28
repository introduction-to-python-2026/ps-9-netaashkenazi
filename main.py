

import pandas as pd
import seaborn as sns
parkinsons= pd.read_csv('parkinsons.csv')
y = parkinsons[ 'status']
x = parkinsons[ ['PPE', 'DFA']]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors= 1)
model.fit(x_train, y_train)
from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print(accuracy)
