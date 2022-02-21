# Regularized Regression Explorations
# Code by Pascal Wallisch and Stephen Spivack
# Date: 10-20-21

#%% 1. Make the data for the multiple regression

# Create Sata:
import numpy as np
n = 200
satM = 500 + np.random.normal(0,1,n) * 111    

# Compute descriptives:
satMean = np.mean(satM)
satMedian = np.median(satM)
satMin = np.min(satM)
satMax = np.max(satM)

# Compute correlation coefficient:
Y = satM + 200 * np.random.normal(0,1,n)
Y = np.round(Y/10)
X = np.round(satM)
r = np.corrcoef(X,Y)

# Do that again on more sata:
x = satM/200
Y2 = np.log(x) + 0.5 * np.random.normal(0,1,n)
r2 = np.corrcoef(x,Y2)

# Reformat Y2:
Y2 = np.round(Y2*21 + 20)

# Run regression:
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(satM.reshape(len(satM),1),Y2)
b0, b1 = model.intercept_, model.coef_
yHat = b1 * satM + b0

# Plot data:
import matplotlib.pyplot as plt
plt.plot(satM,Y2,'o',markersize=3)
plt.xlabel('Math SAT score')
plt.ylabel('Class grade')
plt.plot(satM,yHat,color='orange',linewidth=0.5)

#%% 2. Overfitting example

Q = np.array([[satM[7],Y2[7]],[satM[180],Y2[180]]]) # pick 2 points
model = LinearRegression().fit(Q[:,0].reshape(len(Q),1),Q[:,1]) # fit model
b0, b1 = model.intercept_, model.coef_ # extract betas
yHat = b1 * satM + b0 # build model
plt.plot(Q[:,0],Q[:,1],'o',markersize=3) # plot data
plt.xlabel('Math SAT score')
plt.ylabel('Class grade')
plt.plot(satM,yHat,color='orange',linewidth=0.5) # plot line
r_sq = model.score(Q[:,0].reshape(len(Q),1),Q[:,1])
print(r_sq) # Captures 100% of the variance

#%% Enter ridge regression
# Linear least squares with L2 regularization

from sklearn.linear_model import Ridge
lam = np.linspace(0,5.5,56) # 0 to 5.5 in 0.1 increments
for ii in range(len(lam)):
    model = Ridge(alpha=lam[ii])
    model.fit(Q[:,0].reshape(-1,1),Q[:,1])
    b0, b1 = model.intercept_, model.coef_ 
    x = np.array([200,400])
    yHat = b1 * x + b0
    print(yHat)
    plt.plot(x,yHat) 
    plt.pause(0.01)

#%% 3. Let's do the multiple regression version

# Load data:
x = np.genfromtxt('mRegDataX.csv',delimiter=',') # satM satV hoursS gpa appreciation fearM fearT
y = np.genfromtxt('mRegDataY.csv',delimiter=',') # outcome: class score

# 2. Doing the full model and calculating the yhats:
model = LinearRegression().fit(x,y)
b0, b1 = model.intercept_, model.coef_
y_hat = np.dot(b1,x.transpose()) + b0

# 3. Scatter plot between predicted and actual score of full model:
r = np.corrcoef(y_hat,y)
plt.plot(y_hat,y,'o',markersize=5)
plt.xlabel('Predicted grade score')
plt.ylabel('Actual grade score')
plt.title('R: {:.3f}'.format(r[0,1])) 

# 4. Splitting the dataset for cross-validation:
x1 = np.copy(x[0:100,:])
y1 = np.copy(y[0:100])
model = LinearRegression().fit(x1,y1)
b0_1, b1_1 = model.intercept_, model.coef_

x2 = np.copy(x[100:,:])
y2 = np.copy(y[100:])
model = LinearRegression().fit(x2,y2)
b0_2, b1_2 = model.intercept_, model.coef_

# 5. Cross-validation. Using the betas from the first (training) dataset, but
# measuring the error with the second (test) dataset
y_hat1 = np.dot(b1_1,x1.transpose()) + b0_1
y_hat2 = np.dot(b1_1,x2.transpose()) + b0_1
rmse = np.sqrt(np.mean((y_hat2 - y2)**2))

#%% 4. Using ridge regression to find optimal lambda: a scikit-learn implementation

# Load libraries:
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Init parameters:
xTrain, xTest, yTrain, yTest = train_test_split(x, y.reshape(-1,1), test_size=0.2, random_state=0)
lambdas = np.linspace(-10,10,2001)
cont = np.empty([len(lambdas),2])*np.NaN # [lambda error]
for ii in range(len(lambdas)):
    ridgeModel = Ridge(alpha=lambdas[ii]).fit(xTrain, yTrain)
    cont[ii,0] = lambdas[ii]
    error = mean_squared_error(yTest,ridgeModel.predict(xTest),squared=False)
    cont[ii,1] = error

plt.plot(cont[:,0],cont[:,1])
plt.xlabel('Lambda')
plt.ylabel('RMSE')
plt.title('Ridge regression')
plt.show()
print('Optimal lambda:',lambdas[np.argmax(cont[:,1]==np.min(cont[:,1]))])

#%% 5. Now do the same thing--but with lasso regression

# Load libraries:
from sklearn.linear_model import Lasso

# Init parameters:
xTrain, xTest, yTrain, yTest = train_test_split(x, y.reshape(-1,1), test_size=0.2, random_state=0)
lambdas = np.linspace(-10,10,2001)
cont = np.empty([len(lambdas),2])*np.NaN # [lambda error]

for ii in range(len(lambdas)):
    ridgeModel = Lasso(alpha=lambdas[ii]).fit(xTrain, yTrain)
    cont[ii,0] = lambdas[ii]
    error = mean_squared_error(yTest,ridgeModel.predict(xTest),squared=False)
    cont[ii,1] = error

plt.plot(cont[:,0],cont[:,1])
plt.xlabel('Lambda')
plt.ylabel('RMSE')
plt.title('Lasso regression')
plt.show()
print('Optimal lambda:',lambdas[np.argmax(cont[:,1]==np.min(cont[:,1]))])
