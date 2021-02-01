import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import *

data=pd.read_csv(r'C:\Users\Gobu\OneDrive\Desktop\General\data\train\Kaggle\train.csv')
data2 = pd.read_csv(r'C:\Users\Gobu\OneDrive\Desktop\General\data\test\Kaggle\test.csv')

#Independent variable
nm = list(data.columns.values)
nm.remove("SalePrice")
nm2 = list(data2.columns.values)
for x in nm:
    data[x].fillna(0, inplace = True) 
    for y in data[x]:
        if "str" in str(type(y)):
            data[x].replace({y:0}, inplace=True)

x = data[nm]

nm2 = list(data2.columns.values)
for x2 in nm2:
    data2[x2].fillna(0, inplace = True) 
    for y2 in data2[x2]:
        if "str" in str(type(y2)):
            data2[x2].replace({y2:0}, inplace=True)

x2 = data2[nm2]

y=data.loc[:,'SalePrice']

"""
print(x2)
print(type(x2))

print(x)
print(type(x))
"""
#split train - test data
x_train = x
y_train = y

x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,train_size=0.8,random_state=1)

#train model
regr=linear_model.LinearRegression()
regr.fit(x_train,y_train)

#test model
y_pred=regr.predict(x2)
print(y_pred)

y_pred = list(y_pred)

listt = {}
listt["Id"] = x2["Id"].tolist()
listt["SalePrice"] = y_pred
print(' ')
print(listt)

sub_csv=pd.DataFrame(listt, columns=["Id", "SalePrice"])

sub_csv.to_csv("sub_csv.csv", index=False)

