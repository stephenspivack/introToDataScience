# Simple Linear Regression
# Code by Pascal Wallisch and Stephen Spivack
# Date: 02-21-22

#%% 1. Can we predict annual income from IQ?

# Load data:
import numpy as np
data = np.genfromtxt('iqIncome.csv',delimiter=',') # Column 1 = IQ, 2 = income

# Compute descriptives:
D1 = np.mean(data,axis=0) # take mean of each column
D2 = np.median(data,axis=0) # take median of each column
D3 = np.std(data,axis=0) # take std of each column
D4 = np.corrcoef(data[:,0],data[:,1]) # pearson r = 0.44

# Remember: we know that for IQ the population distribution is mu = 100 and 
# sigma = 15. Confirm that this is true, for our sample.

# Plot data:
import matplotlib.pyplot as plt 
plt.plot(data[:,0],data[:,1],'o',markersize=1) 
plt.xlabel('IQ') 
plt.ylabel('income')  

#%% 2. Computing simple linear regression by hand

# Initialize container:
regressContainer = np.empty([len(data),5]) 
regressContainer[:] = np.NaN 

# Compute each component for our SLR equation:
for ii in range(len(data)):
    regressContainer[ii,0] = data[ii,0] # IQ
    regressContainer[ii,1] = data[ii,1] # income
    regressContainer[ii,2] = data[ii,0]*data[ii,1] # IQ * income
    regressContainer[ii,3] = data[ii,0]**2 # IQ squared
    regressContainer[ii,4] = data[ii,1]**2 # income squared
    
# Compute m ("slope"):
mNumerator = len(data)*sum(regressContainer[:,2]) - sum(regressContainer[:,0])*sum(regressContainer[:,1])
mDenominator = len(data)*sum(regressContainer[:,3]) - (sum(regressContainer[:,0]))**2
m = mNumerator/mDenominator

# Compute b ("y-intercept"):
bNumerator = sum(regressContainer[:,1]) - m * sum(regressContainer[:,0])
bDenominator = len(data)
b = bNumerator/bDenominator
rSquared = D4[0,1]**2 

# Add regression line to visualization:
plt.plot(data[:,0],data[:,1],'o',markersize=1) 
plt.xlabel('IQ') 
plt.ylabel('income')  
yHat = m*data[:,0] + b # slope-intercept form, y = mx + b
plt.plot(data[:,0],yHat,color='orange',linewidth=0.5) # orange line for the fox
plt.title('By Hand: R^2 = {:.3f}'.format(rSquared)) # add title, r-squared rounded to nearest thousandth

#%% 3. Now let's use scikit-learn to do the same thing

# Import scikit learn:
from sklearn.linear_model import LinearRegression

# Initialize data:
x = data[:,0].reshape(len(data),1) 
y = data[:,1] 

# Create and fit model:
model = LinearRegression().fit(x,y)

# Evaluate model and plot data:
rSqr = model.score(x,y) # Note that this is the same as by hand!
slope = model.coef_ # Same goes for B1 (slope)
intercept = model.intercept_ # And B0 (intercept)
yHat = slope * data[:,0] + intercept

plt.plot(data[:,0],data[:,1],'o',markersize=1) 
plt.xlabel('IQ') 
plt.ylabel('income')  
plt.plot(data[:,0],yHat,color='orange',linewidth=0.5)
plt.title('Using scikit-learn: R^2 = {:.3f}'.format(rSqr))
