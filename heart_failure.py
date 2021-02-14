import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import *

#load data
data=pd.read_csv(r'C:\Users\Gobu\OneDrive\Desktop\heart_failure_clinical_records_dataset.csv')

nms = []

for col in data:
    nms.append(col)

print(nms)

"""
#independent variable
x=data[['bedrooms','sqft_living','sqft_lot', 'sqft_above', 'sqft_basement','sqft_living15', 'sqft_lot15']]

#dependent variable
y=data.loc[:,'price']

#split train - test data
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,train_size=0.8,random_state=1)

#train model
regr=linear_model.LinearRegression()
regr.fit(x_train,y_train)

#test model
y_pred=regr.predict(x_test)
print(y_pred)
"""
