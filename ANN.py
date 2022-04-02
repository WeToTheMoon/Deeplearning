import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('C:/Users/kesch/OneDrive/Documents/Deeplearning/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
Y = dataset.iloc[:, -1].values

le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

fs = StandardScaler()
X = fs.fit_transform(X) 
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

nn = tf.keras.models.Sequential()
nn.add(tf.keras.layers.Dense(units=6, activation='relu'))
nn.add(tf.keras.layers.Dense(units=6, activation='relu'))
nn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
nn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
nn.fit(np.array(x_train), np.array(y_train), batch_size=32, epochs = 100)

y_pred = nn.predict(x_test)
y_pred = (y_pred > 0.5)
print(accuracy_score(y_test, y_pred))