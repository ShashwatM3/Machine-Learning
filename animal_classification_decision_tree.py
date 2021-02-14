import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import *
from sklearn.tree import DecisionTreeClassifier, plot_tree

#load data
data=pd.read_csv(r'C:\Users\Gobu\OneDrive\Desktop\zoo.csv')
print(data.head())

label_encoder = preprocessing.LabelEncoder()
data['animal_name']= label_encoder.fit_transform(data['animal_name'])

nms = []

for col in data.columns: 
    nms.append(col)

#independent variable
x=data[nms]

#dependent variable
y=data.loc[:,'animal_name']

#split train - test data
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,train_size=0.8,random_state=1)

#train model
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

#test model
y_pred=clf.predict(x_test)
print(label_encoder.inverse_transform(y_pred))
