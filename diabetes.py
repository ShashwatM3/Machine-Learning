# import modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


#load data
data_set=pd.read_csv('salary_data.csv')

# display first 5 rows
print(data_set.head())
print(data_set)

#dependent variable
y=data_set.loc[:,'Salary']
y=np.array(y)
print(y)

#split train & test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

#Create Linear Regression Object
regr = linear_model.LinearRegression()

#Train Model
regr.fit(x_train,y_train)

#Getting Predictions
y_pred=regr.predict(x_test)
print(y_pred)

# graphs
plt.scatter(x_test,y_test,color='green')
plt.plot(x_test,y_pred,color='pink',linewidth=3)
plt.xticks(())
plt.yticks(())
