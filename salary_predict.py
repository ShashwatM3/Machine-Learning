# import modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#load data
data_set=pd.read_csv(r'C:\Users\Gobu\OneDrive\Desktop\General\data\Salary_Data.csv')

#independent variable
x=data_set.loc[:,'YearsExperience']
x=np.array(x)
x=x.reshape(-1,1)

#dependent variable
y=data_set.loc[:,'Salary']
y=np.array(y)

#split train & test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# create object
regr = linear_model.LinearRegression()

#train model
regr.fit(x_train,y_train)

test = input("Enter years of experience - ")
test_list = [[]]
test_list[0].append(float(test))
print(test_list)

# getting predictions
y_pred=regr.predict(test_list)
print("Predicted salary - "+str(y_pred))
