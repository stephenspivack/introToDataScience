# Descriptive Stats
# Code by Pascal Wallisch and Stephen Spivack
# Date: 02-21-22

#%% Today, we are going to give you a basic overview of numpy functions that
# compute measures of central tendency. That includes the mean, median and
# mode. We will also look at the dispersion measures standard deviation and
# mean absolute deviation, both using Numpy and simulations.

#%% 1. Mean and Median using Numpy

# import the libraries:
import numpy as np 
import matplotlib.pyplot as plt

# simulate 10 random samples of n = 10000 from a standard normal distribution:
numSamples = 10
sampleSize = int(1e4)
sata1 = np.random.normal(0,1,[sampleSize,numSamples]) # mean is 0, std is 1, size is 10000x10

# compute mean across each column:
columnMean = np.mean(sata1,axis=0)
columnMedian = np.median(sata1,axis=0)

# bar plot:
plt.subplot(1,2,1)
plt.bar(np.linspace(1,numSamples,numSamples), columnMean)
plt.title('Mean')
plt.subplot(1,2,2)
plt.bar(np.linspace(1,numSamples,numSamples), columnMedian)
plt.title('Median')

# You can also compute across rows by setting axis = 1:
rowMean = np.mean(sata1,axis=1)
rowMedian = np.median(sata1,axis=1)

# default takes mean/median across entire matrix and returns one value:
grandMean = np.mean(sata1)
grandMedian = np.median(sata1)

#%% 2. Mode using Numpy

# simulate 1 random sample of n = 10000 from a poisson distribution:
sata2 = np.random.poisson(5,sampleSize) # lambda = 5

# compute the mode:
theMode = np.argmax(np.bincount(sata2)) 
# count number of occurrences of each value in array of non-negative ints;
# then return the index of the maximum value.

# plot:
plt.hist(sata2,bins=len(np.bincount(sata2))) 
plt.title('Poisson distribution for lambda = 5 and n = 10000')
plt.plot([theMode,theMode],[0,2000],color='red',linewidth=1)

#%% 3. Standard deviation using Numpy

# compute std across each column:
columnStd = np.std(sata1,axis=0)

# compute std across each row:
rowStd = np.std(sata1,axis=1)

# compute std across entire matrix:
grandStd = np.std(sata1) 

#%% 4. Custom function to compute MAD
# Note: saved as an external file -> mean_absolute_deviation.py
# Make sure this file is in same file path as this script so you can run the
# simulations below.

def mean_absolute_deviation_func(data):
    M = np.mean(data)
    sum = 0
    for ii in range(len(data)):
        dev = np.absolute(data[ii] - M)
        sum = sum + dev
    mad = sum/len(data)
    return mad # you mad bro?

#%% 5. Simulation - Mean Absolute Deviation (MAD) vs. Standard Deviation (SD): 
# This is a simulation, as seen in lecture

# Import MAD function:
from mean_absolute_deviation import mean_absolute_deviation_func # from file (no .py) import function

# Preallocate variables:
numElements = int(1e4)
numIterations = 30
M = np.zeros([numElements,numIterations])
S = np.zeros([numElements,numIterations])

# Compute MAD and SD:
for jj in range(numIterations):
    X = np.random.normal(0,1,[numElements,jj+2]) # (mean, sd, [number of draws, number of columns])
    for ii in range(numElements):
        data = X[ii,:]
        M[ii,jj] = mean_absolute_deviation_func(data) # custom M.A.D. function
        S[ii,jj] = np.std(data) #Take the standard deviation, using the numpy function
        
# Plot data:
for ii in range(numIterations):
    # Subplot 1:
    plt.subplot(1,2,1)
    plt.plot(M[:,ii],S[:,ii],'o',markersize=.5)
    maxi = np.max(np.array([M[:,ii],S[:,ii]]))
    line = np.array([0,maxi])
    plt.plot(line,line,color='red',linewidth=0.5)
    plt.title('N = {}'.format(ii+2))
    plt.xlabel('MAD')
    plt.xlim([0,maxi])
    plt.ylabel('SD')  
    plt.ylim([0,maxi])
    # Subplot 2:
    plt.subplot(1,2,2)
    plt.hist(S[:,ii]/M[:,ii],100)
    meanToPlot = np.mean(S[:,ii]/M[:,ii]) #Take the mean, using the numpy function
    medianToPlot = np.median(S[:,ii]/M[:,ii]) #Take the median, using the numpy function
    plt.title('Mean = {:.3f}'.format(meanToPlot) + ', Median = {:.3f}'.format(medianToPlot))
    plt.xlabel('SD/MAD')
    plt.xlim(1,2)
    plt.ylabel('Relative Bin Count')
    ax = plt.gca()
    ax.axes.yaxis.set_ticks([])
    # Timelapse between each plot:
    plt.pause(.01)

#%% 6. A closer look at MAD vs. SD
# Given a normal distribution, the MAD to SD ratio is 0.8
# This also means that the SD to MAD ratio is 1.25 (as shown in the simulation)
# Let's compute and plot the MAD against the SD for different values of the SD

# Initialize parameters:
numStds = 100 # integer values, 1 to 100
numElements = int(1e5) # number of random numbers drawn per SD

# Compute MAD for each SD (1 to 100):
meanAbsDev = np.zeros(numStds) # initialize array of zeros to store MAD
for ii in range(numStds): # loop through each SD value
     data = np.random.normal(0,ii+1,numElements) # draw numElements random numbers from a normal distribution
     meanAbsDev[ii] = mean_absolute_deviation_func(data) # then compute MAD

# Print ratio of MAD to SD:
print('Ratio: ',np.mean(meanAbsDev/np.linspace(1,numStds,numStds)))
      
# Line plot of MAD against SD:
plt.plot(meanAbsDev)
plt.ylabel('MAD')
plt.xlabel('SD')
plt.title('MAD vs. SD for normal distributions')
