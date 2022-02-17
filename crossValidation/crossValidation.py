# Overfitting and cross-validation
# Code by Pascal Wallisch and Stephen Spivack
# Date: 02-16-22

#%% Overfitting
# Modeling error that occurs when a function fits the set of data points too well
# 1. Sample points from a curve
# 2. Fit a polynomial to those points until we overfit
# 3. Leave one out cross-validation to demonstrate the escalation of RMSE when overfitting

# Import libraries:
import numpy as np
import matplotlib.pyplot as plt

# Initialize parameters:
noiseMagnitude = 2 # how much random noise is there?
numData = 8 # how many measurements (samples) of the signal?
numPoints = 1001 
leftRange = -5 
rightRange = 5
x = np.linspace(leftRange,rightRange,numPoints) # determine the location 
# of evenly spaced points from -5 to 5 to use as an x-base

# Determine the functional relationship between x and y in reality (ground truth):
sig = 1 # user determines whether the signal is quadratic (1) or cubic (2)
if sig == 1:
    y1 = x**2 # quadratic function
elif sig == 2:
    y1 = x**3 # cubic function
    
# Compute signal plus noise:
y = y1 + noiseMagnitude * np.random.normal(0,1,numPoints) # signal + noise

# Plot data:
plt.figure(1)
plt.plot(x,y1,color='blue',linewidth=5)
plt.xlabel('X') 
plt.ylabel('Y')  
plt.title('Ground Truth')

plt.figure(2)
plt.plot(x,y,color='blue',linewidth=1)
plt.xlabel('X') 
plt.ylabel('Y')  
plt.title('Signal plus noise')

#Ground truth with noise in one plot
plt.figure(3)
plt.plot(x,y,color='blue',linewidth=1)
plt.plot(x,y1,color='black',linewidth=5)

#%% Determine the location of the sampling (measuring) points 

# Randomly draw points to sample:
samplingIndices = np.random.randint(1,numPoints,numData) # random points, from anywhere on the signal

# Plot data as a subsample of the noisy signal:
plt.plot(x,y,color='blue',linewidth=1)
plt.plot(x,y1,color='black',linewidth=5)
plt.plot(x[samplingIndices],y[samplingIndices],'o',markersize=4,color='red')
plt.xlim(-5,5) # keep it on the same x-range as before

# Note: Parabola doesn't fit perfectly because there is noise (measurement error). We are
# overfitting to noise. The more noise, the worse this effect is
# In real life, all measurements are contaminated with noise, so overfitting
# to noise is always a concern.

#%% (Over)fitting successive polynomials and calculating RMSE at each point

rmse = np.array([]) # capture RMSE for each polynomial degree
for ii in range(numData): # loop through each sampling point
    plt.subplot(2,4,ii+1)
    numDegrees = ii+1 # degree of polynomial to fit to our 8 data points
    p = np.polyfit(x[samplingIndices],y[samplingIndices],numDegrees) # returns a vector of coefficients p that minimizes the squared error
    yHat = np.polyval(p,x) # evaluate the polynomial at specific values
    plt.plot(x,yHat,color='blue',linewidth=1)
    plt.plot(x[samplingIndices],y[samplingIndices],'ro',markersize=3)
    error = np.sqrt(np.mean((y[samplingIndices] - yHat[samplingIndices])**2))
    plt.title('Degrees: {}'.format(numDegrees) + ', RMSE = {:.3f}'.format(error))
    rmse = np.append(rmse,error) # keep track of RMSE - we will use this later
    
#%% Plotting RMSE of the training set as function of polynomial degree
plt.plot(np.linspace(1,numData,numData),rmse)
plt.title('Apparent RMSE as a function of degree of polynomial')
plt.xlabel('Degree of polynomial')
plt.ylabel('RMSE')

#%% Leave-one-out procedure to cross-validate the number of terms in the
# model. Note: We randomly pick one of the test points to use to calculate
# the RMSE with. We use the other data points to fit the model
# This method is called "leave one out" and is very computationally
# expensive, as one has to fit the model n-1 times

# Initialize parameters:
numRepeats = 100 # Number of samples - how often are we doing this?
rmse = np.zeros([numRepeats,numData-1]) # Reinitialize RMSE (100x7)
# For each polynomial degree, 100x we are going to randomly pick one of
# the points from the set of 8 and compute the RMSE
# We are then going to fit the model from the remaining (7) points
# This is why we only go up to the 7th degree polynomial

# Compute RMSE on test set:
for ii in range(numRepeats): # Loop from 0 to 99
    testIndex = np.random.randint(0,numData,1) # Randomize test index - pick randint from 0 to 7
    testSet = samplingIndices[testIndex] # Find the test set (= 1 random data point from our 8)
    trainingSet = np.copy([samplingIndices]) # Make copy of sampling indices
    trainingSet = np.delete(trainingSet,testIndex) # Delete the test subset
    for jj in range(numData-1): # Loop from 0 to 6 - for each poly degree
        numDegrees = jj+1 # degrees are from 1 to 7, so add 1 to jj each time
        p = np.polyfit(x[trainingSet],y[trainingSet],numDegrees) # compute coefficients
        yHat = np.polyval(p,x)  # then evaluate
        # Calculate RMSE with the test set (just the single point we randomly chose above):
        rmse[ii,jj] = np.sqrt(np.mean((y[testSet] - yHat[testSet])**2)) # store this in rmse container

# Plot data:
plt.plot(np.linspace(1,numData-1,7),np.mean(rmse,axis=0))
plt.title('Real RMSE as a function of degree of polynomial')
plt.xlabel('Degree of polynomial')
plt.ylabel('RMSE measured only at points left out from building model')   

# The solution? Where RMSE is minimal
solution = np.amin(np.mean(rmse,axis=0)) # value
index = np.argmin(np.mean(rmse,axis=0)) # index
print('The RMSE is minimal at polynomial of degree: {}'.format(index+1)) 

# Note - the console will give you warnings that the polyfit is poorly conditioned sometimes. 
# That's another dead giveaway that you are overfitting. Too many parameters, not enough data.

#%% Revisiting MLR using cross-validation
# Predicting class performance from SAT math, SAT verbal, hours studied, GPA, 
# appreciation of statistics, fear of math, and fear of teacher
# There are 200 students in this case.

# Load libraries:
from sklearn.linear_model import LinearRegression

# Load data:
x = np.genfromtxt('mRegDataX.csv',delimiter=',') # satM satV hoursS gpa appreciation fearM fearT
y = np.genfromtxt('mRegDataY.csv',delimiter=',') # outcome: class score

# Doing the full model and calculating the yhats:
model = LinearRegression().fit(x,y)
b0, b1 = model.intercept_, model.coef_
yHat = b1[0]*x[:,0] + b1[1]*x[:,1] + b1[2]*x[:,2] + b1[3]*x[:,3] + b1[4]*x[:,4] + b1[5]*x[:,5] + b1[6]*x[:,6] + b0

# Scatter plot between predicted and actual score of full model:
plt.plot(yHat,y,'o',markersize=5)
plt.xlabel('Predicted grade score')
plt.ylabel('Actual grade score')
rSqr = model.score(x,y)
plt.title('R^2 = {:.3f}'.format(rSqr))

# Splitting the dataset (here, 50-50) for cross-validation:
x1 = np.copy(x[0:100,:])
y1 = np.copy(y[0:100])
regr = LinearRegression().fit(x1,y1) 
betas1 = regr.coef_ 
yInt1 = regr.intercept_

x2 = np.copy(x[100:,:])
y2 = np.copy(y[100:])
regr = LinearRegression().fit(x2,y2) 
betas2 = regr.coef_ 
yInt2 = regr.intercept_

# 5) Cross-validation. Using the betas from the first (training) dataset, but
#    measuring the error with the second (test) dataset
yHat1 = np.dot(x1,betas1) + yInt1
yHat2 = np.dot(x2,betas1) + yInt1
rmse = np.sqrt(np.mean((yHat2 - y2)**2))
print('RMSE:',rmse)
