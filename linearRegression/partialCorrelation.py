# Partial Correlation
# Code by Pascal Wallisch and Stephen Spivack
# Date: 02-21-22

#%% Partial correlation

# We want to know the relationship between income and violent incidents.
# Lead levels are a known confounding variable, so we want to control for 
# this.

# Logic: do 2 simple linear regressions for lead levels vs. income and lead 
# levels vs. violent incidents, then correlate the residuals; this is the 
# partial correlation.

# Import libraries:
import numpy as np
from sklearn.linear_model import LinearRegression

# Load data: income, violent incidents, lead levels
data = np.genfromtxt('leadIncome.csv',delimiter=',')

# Compute descriptives: 
D1 = np.mean(data,axis=0) # take mean of each column
D2 = np.median(data,axis=0) # take median of each column
D3 = np.std(data,axis=0) # take std of each column

# Compute correlation between income and violence:
r = np.corrcoef(data[:,0],data[:,1]) 
print('Pearson correlation:',np.round(r[0,1],3))

# Initialize data for first SLR:
x = data[:,2].reshape(len(data),1)  # lead levels
y = data[:,0] # income

# Create and fit model:
model = LinearRegression().fit(x, y)

# Compute residuals:
slope = model.coef_ # Same goes for B1 (slope)
intercept = model.intercept_ # And B0 (intercept)
yHat = slope * x + intercept
residuals1 = y - yHat.flatten()

# Initialize data for second SLR:
x = data[:,2].reshape(len(data),1)  # lead levels
y = data[:,1] # violent incidents

# Create and fit model:
model = LinearRegression().fit(x, y)

# Compute residuals:
slope = model.coef_ # Same goes for B1 (slope)
intercept = model.intercept_ # And B0 (intercept)
yHat = slope * x + intercept
residuals2 = y - yHat.flatten()

# Correlate the residuals:
partCorr = np.corrcoef(residuals1,residuals2)
print('Partial correlation:',np.round(partCorr[0,1],3))
    