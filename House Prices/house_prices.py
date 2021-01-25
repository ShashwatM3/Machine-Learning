import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import *

data=pd.read_csv(r'C:\Users\Gobu\OneDrive\Desktop\kc_house_data.csv')
print(data.head())

#Independent variable
x=data[['bedrooms','sqft_living','sqft_lot', 'sqft_above', 'sqft_basement','sqft_living15', 'sqft_lot15']]
print(type(x))

#dependent variable
y=data.loc[:,'price']
print(type(y))

"""
#split train - test data
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,train_size=0.8,random_state=1)

#train model
regr=linear_model.LinearRegression()
regr.fit(x_train,y_train)

#test model
y_pred=regr.predict(x_test)
print(y_pred)

# graphs
plt.scatter(x_test['bedrooms'],y_test,color='orange')
plt.scatter(x_test['bedrooms'],y_pred,color='blue')
plt.xticks(())
plt.yticks(())
"""
