# Correlation
# Code by Pascal Wallisch and Stephen Spivack
# Date: 02-20-22

#%% Logic of correlation:
# 1. Start with equation for standard devitation
# 2. Square both sides to get variance
# 3. Distribute numerator and replace one Xi with Yi to get covariance
# 4. Divide by product of Xstd and Ystd to get correlation

#%% 1. Computing correlations with Numpy

# Import libraries:
import numpy as np
import matplotlib.pyplot as plt

# Simulate 2 data sets:
numElements = 100
noiseFactor = 5
noise = noiseFactor * np.random.normal(0,1,numElements)
data1 = np.linspace(1,numElements,numElements) + noise
data2 = np.linspace(1,numElements,numElements) + noiseFactor * noise

# Compute correlation coefficient:
r = np.corrcoef(data1,data2)

# Plot data:
plt.plot(data1,data2,'o')
plt.xlabel('Data 1')
plt.ylabel('Data 2')
plt.title('r = {:.3f}'.format(r[0,1]))

#%% 2. Correlation simulation (aleatory calculations)
# Aleatory = "depending on the throw of a dice or on chance; random"

# 0. Import libraries
from scipy import stats # We need scipy to compute Spearman's Rho (remember: numpy and matplotlib are already imported)
    
# 1. Initialize parameters:
numReps = 1000 # Number of experiment repeats (to see the average correlation)
numGears = 100 # We are looping through 1 to 100 gears
m = 10 # Number of mutually exclusive events (gear slots)
empVsExp = np.empty([numGears+1,4]) # Initialize container to store correlations: perfect world and simulation
empVsExp[:] = np.NaN # Convert to NaN
counter = 0 # Initialize counter
  
# 2. Run simulation:  
for c in range(numGears+1): # Loop through each gear (from 0 to 100)
    
    # Simulate aleatory observations:
    ratio = c/numGears # relative to total - proportion of gears in 2nd observation that change
    observation1 = np.random.randint(m+1,size=(numGears,numReps)) # first observation; m slots (mutually exclusive events) per gear
    observation2 = np.copy(observation1) # second observation - same as first
    observation2[0:c,:] = np.random.randint(m+1,size=(c,numReps)) # randomly change c gears in second observation
    
    # Compute Pearson R for each repeat:
    temp = np.empty([numReps,1]) # initialize empty container to store each r value
    temp[:] = np.NaN # convert to NaN
    for i in range(numReps): # Loop through each experimental repeat
        r = np.corrcoef(observation1[:,i],observation2[:,i]) # compute the Pearson R
        temp[i] = r[0,1] # store coefficient in temp variable
    
    # Compute Spearman Rho for each repeat:
    temp2 = np.empty([numReps,1]) # initialize empty container to store each rho value
    temp2[:] = np.NaN # convert to NaN
    for i in range(numReps): # Loop through each experimental repeat 1 to 1000
        r = stats.spearmanr(observation1[:,i],observation2[:,i]) # Compute Spearman Rho
        temp2[i] = r[0] # store coefficient in temp2 variable
    
    # Store data:
    empVsExp[counter,0] = ratio
    empVsExp[counter,1] = np.mean(temp) # take mean R for all experimental repeats
    empVsExp[counter,2] = 1 - ratio
    empVsExp[counter,3] = np.mean(temp2) # take mean rho for all experimental repeats
    counter = counter + 1 # Increment the counter
    
    # Plot data:
    plt.plot(sum(observation1),sum(observation2),'o',markersize=.75)
    plt.title('Ratio = {:.2f}'.format(empVsExp[c,0]) + ', r = {:.3f}'.format(empVsExp[c,1]) + ', rho = {:.3f}'.format(empVsExp[c,3]))
    plt.xlim(300,700)
    plt.ylim(300,700)
    plt.pause(.01) # pause (in seconds) between iterations
    
#%% Ratio of R vs. Rho as a function of increasing correlation:
ascendingMatrix = np.flipud(np.copy(empVsExp)) # copy array and flip upside down
plt.plot(ascendingMatrix[:,1]/ascendingMatrix[:,3]) # Hint: It stabilizes - 
# run 2 multiple times to confirm this. More unstable at lower correlations, as ranks will jump more
plt.title('Ratio of r vs. rho as a function of increasing correlation')

#%% Correlation simulation 2 - What is going on with the spearman correlation by using random ranks
# Instead of using scipy let's manually compute Spearman's Rho. By hand. To understand it. Just this once.


# 1. Initialize variables:
scaleFactor = 6 # Try different ones to determine that it has to be 6 to map from -1 to 1
numReps = int(1e4) # Number of repetitions
maxRanks = 50
data = np.empty([maxRanks-1,numReps,5]) # Initialize the 3D array (#stack,#rows,#columns) to put the data here
                                        # ranks are from 2 to 50, for a total of 49 unique ranks
data[:] = np.NaN # Convert to NaN
counter = 0

# 2. Run simulation:
for r in range(2,maxRanks+1): # Number of ranks involved - this increases by 1 for each iteration
    for i in range(numReps): # Loop through each rep of a given rank - 10000 reps per r
        temp = np.random.permutation(r) # Randomly permute the ranks (shuffes numbers from 2 to r)
        temp2 = np.random.permutation(r) # Do that again
        d = temp - temp2 # Calculate the rank differences
        dS = d**2 # Square the rank differences. If negative, this yields a sequence of odd squares
        Sd = sum(dS) # Sum the squared rank differences
        numerator = scaleFactor * Sd # Mulitply by the scale factor -> numerator
        denominator = r*(r**2-1)  # Play around with not squaring it, or go +1 instead of -1
        pos = numerator/denominator # How large is the positive part - if it is larger than 1, correlation will be negative
        rho = 1-pos
        # Store data:
        data[counter,i-1,0] = Sd
        data[counter,i-1,1] = numerator
        data[counter,i-1,2] = denominator
        data[counter,i-1,3] = pos
        data[counter,i-1,4] = rho
    counter = counter + 1 # Increment counter to keep track of the stack

# 3. Plot data:
counter = 0 # initialize counter
meanAbsValue = np.empty([maxRanks-1,1]) # this is where we store abs of all rhos for a given r
meanAbsValue[:] = np.NaN # make sure to convert to NaN
for r in range(2,maxRanks+1): # Loop through each stack (2 to 50)
    tempData = data[counter,:,:] # take the stack that corresponds to r
    meanAbsValue[counter] = np.mean(abs(tempData[:,4])) # take the mean abs value of all rhos
    plt.hist(tempData[:,4],bins=51) # plot histogram and specify bin count
    plt.title('Number of ranks involved: {}'.format(r)) # add title
    plt.pause(0.1) # pause (in seconds) between iterations
    plt.xlim([-1,1]) # If you try different scale factors, you have to expand this too
    counter = counter + 1 # Increment counter to keep track of the stack


#%% Average correlation *magnitude* (regardless of sign) as a function of numbers involved
# This is the correlation *magnitude* you can expect if you just draw random
# numbers and correlate them, as a function of the numbers involved. This
# has to be taken into account when assessing any correlation. Because this
# is the effect of chance. It is easy to see why it would be 1 for 2
# numbers, and if they are ranks, because it is either 1 2 vs. 1 2 (rho = 1)
# or 1 2 vs. 2 1 (rho = -1). It's a line.
plt.plot(meanAbsValue)   
plt.title('Correlation magnitude expected by chance as a function of the number of pairs')
plt.xlabel('Number of pairs involved in the correlation')
plt.ylabel('Correlation magnitude')


#%% 4. A linear algebra view on correlation
# As you now know, the correlation between 2 variables can be interpreted as
# a relationship between elements (the x- and y- variables can be
# interpreted as coordinates in 2D):

mu = 0
sigma = 1
X = np.random.normal(mu,sigma,100)
Y = X + np.random.normal(mu,sigma,100)
temp = np.corrcoef(X,Y)
r = temp[0,1]
plt.plot(X,Y,'o',markersize=.75)
plt.title('r = {:.3f}'.format(r))

# As usual, linear algebra provides an alternative view that is completely
# consistent with the classical view, but can provide enlightening to some.
# In this view, we are looking at the relationship between 2 (two)
# 100-dimensional vectors. For those, the correlation between them is given
# as their dot product. If they are unit vectors. So lets' reduce them to
# unit vectors first. Then take the dot product.

xUnit = X/np.linalg.norm(X) # Convert x to its unit vector
yUnit = Y/np.linalg.norm(Y) # Convert y to its unit vector
rVec = np.dot(xUnit,yUnit) # Take the dot product
print(abs(r-rVec)) # Close enough - within numerical precision

#%% From standard deviation to correlation
# A toy example using numpy and linear algebra

# Import libraries:
import numpy as np
import matplotlib.pyplot as plt

# Init values for X and Y:
n = 100
mu,sigma = 1,2
X = np.random.normal(mu,sigma,n)
Y = X + np.random.normal(mu,sigma,n)

# Compute covariance using numpy:
cov1 = np.cov(X,Y)
print('Covariance from numpy: {:0.3}'.format((cov1)[0,1]))

# Compute covariance using linear algebra:
M = np.concatenate((X.reshape([n,1]),Y.reshape([n,1])),axis=1)
cov2 = np.dot(M.T,M)

# Compute correlation using numpy:
corr1 = np.corrcoef(X,Y)[0,1]
print('Correlation from numpy: {:0.3}'.format(corr1))

# Compute correlation using linear algebra:
X = X/np.linalg.norm(X) # np.linalg.norm computes the magnitude of the input
Y = Y/np.linalg.norm(Y) 
corr2 = np.dot(X,Y)
print('Correlation from linear algebra: {:0.3}'.format(corr2))

# Visualize data:
plt.scatter(X,Y)
plt.title('r = {:0.3}'.format(corr1))  
