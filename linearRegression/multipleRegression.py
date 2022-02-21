# Multiple Linear Regression
# Code by Pascal Wallisch and Stephen Spivack
# Date: 02-21-22

#%% Multiple regression model 1
# Predicting income from IQ, hours worked and years formal education

# Load libraries:
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

# Load data:
data = np.genfromtxt('determinedIncome.csv',delimiter=',') 
# Column features: IQ, hours worked, years education, income

# Desriptives:
D1 = np.mean(data,axis=0) # take mean of each column
D2 = np.median(data,axis=0) # take median of each column
D3 = np.std(data,axis=0) # take std of each column
D4 = np.corrcoef(data[:,0],data[:,1]) # correlate IQ and hours worked

# Model: IQ
x = data[:,0].reshape(len(data),1) 
y = data[:,3]
model = LinearRegression().fit(x,y)
rSqr = model.score(x,y)
print(rSqr)

# Model: IQ and hours worked
x = data[:,:2]
y = data[:,3]
model = LinearRegression().fit(x,y)
rSqr = model.score(x,y)
print(rSqr)

# Model: All factors
x = data[:,:3]
y = data[:,3]
model = LinearRegression().fit(x,y)
rSqr = model.score(x,y)
print(rSqr)
b0, b1 = model.intercept_, model.coef_

# Visualize: actual vs. predicted income (from model)
yHat = np.dot(data[:,:3],b1) + b0
plt.plot(yHat,data[:,3],'o',markersize=.75) 
plt.xlabel('Prediction from model') 
plt.ylabel('Actual income')  
plt.title('R^2 = {:.3f}'.format(rSqr))

#%% Multiple regression model 2
# Predicting class performance from SAT math, SAT verbal, hours studied, GPA, 
# appreciation of statistics, fear of math, and fear of teacher
# There are 200 students in this case.

# Load data:
x = np.genfromtxt('mRegDataX.csv',delimiter=',') # satM satV hoursS gpa appreciation fearM fearT
y = np.genfromtxt('mRegDataY.csv',delimiter=',') # outcome: class score

# Doing the full model and calculating the yhats:
model = LinearRegression().fit(x,y)
b0, b1 = model.intercept_, model.coef_
yHat = np.dot(x,b1) + b0

# Scatter plot between predicted and actual score of full model:
plt.plot(yHat,y,'o',markersize=5)
plt.xlabel('Predicted grade score')
plt.ylabel('Actual grade score')
rSqr = model.score(x,y)
plt.title('R^2 = {:.3f}'.format(rSqr))
