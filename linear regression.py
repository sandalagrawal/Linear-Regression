#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This is First Program to implement Linear Regression
# Author - Dr. Virendra Singh Kushwah
# Problem-1: Implementation of Linear Regression to predict salary of a person
# Step 1 Load Data
import pandas as pd
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, [0]].values #YearsExperience Column
y = dataset.iloc[:,[1]].values #Salary column
print(dataset)
print(X)
print(y)


# In[7]:


# Step 2: Split data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
#test data will be 1/3 and 2/3 will training data
print(X_train) # year as form of training data
print(X_test) # year as form of testing data
print (y_train) # salary - training data
print (y_test) # salary = testing data


# In[8]:


# Step 3: Fit Simple Linear Regression to Training Data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #we are creating object of the linearregression
regressor.fit(X_train, y_train) # we are fitting the model by training data


# In[9]:


# Step 4: Make Prediction
y_pred = regressor.predict(X_test)
# we need to pass testing data value for prediction purpose.
# x_test = year value from the testing data
#y_pred = this will predict the salary according to the year


# In[10]:


# Step 5 - Visualize training set results
import matplotlib.pyplot as plt
# plot the actual data points of training set
plt.scatter(X_train, y_train, color = 'red')
# plot the regression line
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience') #independent variable
plt.ylabel('Salary') #dependent variable
plt.show()


# In[11]:


# Step 6 - Visualize test set results
import matplotlib.pyplot as plt
# plot the actual data points of test set
plt.scatter(X_test, y_test, color = 'red')
# plot the regression line (same as above)
plt.plot(X_test, regressor.predict(X_test), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[12]:


# Step 7 - Make new prediction
new_salary_pred = regressor.predict([[20]])
print('The predicted salary of a person with 15 years experience is ',new_salary_pred)


# In[13]:


# printing values
print('Slope:' ,regressor.coef_)
print('Intercept:', regressor.intercept_)


# In[17]:


import numpy as np
# plot the regression line
# Mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(y)
# Total number of values
n = len(X)
# Using the formula to calculate m and c
numer = 0
denom = 0
for i in range(n):
    numer += (X[i] - mean_x) * (y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
m = numer / denom
c = mean_y - (m * mean_x)
# Print coefficients
print(m, c)
import matplotlib.pyplot as plt
# plot the actual data points of training set
plt.scatter(X, y, color = 'red')
y=m*X+c
plt.plot(X, y, color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience') #independent variable
plt.ylabel('Salary') #dependent variable
plt.show()


# In[18]:


#Finding Residuals or errors
from sklearn import metrics
print ('MSE:',metrics.mean_squared_error(y_test,y_pred))
print ('MAE:',metrics.mean_absolute_error(y_test,y_pred))
print ('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[ ]:




