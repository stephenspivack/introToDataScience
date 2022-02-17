# Resampling Methods
# Code by Pascal Wallisch and Stephen Spivack
# Date: 02-16-22

# This lab presumes that the first 3 sections of the script from last lab
# (inferentialStatistics.py) were run. This includes the loader, the pruner,
# (with the *row-wise* removing of missing values on) and the reformatter 
# (putting ratings from all 3 movies into a the combinedData matrix). 

# Now, we use the same data to implement bootstrapping methods.

#%% 1 Using drawing WITH replacement to estimate confidence intervals 
# (Bootstrapping), by hand 

# Sharing variables across scripts:
import numpy as np 
import matplotlib.pyplot as plt
from inferentialStatistics import combinedData

# Compute mean for each movie:
sampleMeans = np.mean(combinedData,axis=0) 
# These are the real means. We get that straight up from the sample itself.
# So what we do we get from the resampling?

numRepeats = int(1e4) # How many times do we want to resample the 1 empirical 
# sample we have?
nSample = len(combinedData) # Number of data points in the sample

# Preallocate what will contain the bootstrapped sample means:
tychenicMeans = np.empty([numRepeats,1])*np.NaN

# Draw integers - which we'll use as indices, num_repeats times:
tychenicIndices = np.random.randint(0,nSample,[numRepeats,numRepeats])
# Draw random nuumbers from 0 to n_sample (not inclusive), to yield an array
# with dimensions that are num_repeats x num_repeats (10000 x 10000)

# Estimate the stability of the sample mean for which movie?:
whichMovie = 0 # 0 = M1, 1 = M2, 2 = M3 (try all 3)
temp = combinedData[:,whichMovie] # store that data in temp array
for i in range(numRepeats): # loop through each repeat
    tempIndices = tychenicIndices[:,i] # indices for this iteration
    tychenicMeans[i] = np.mean(temp[tempIndices]) # compute the mean

# How good is our estimate of the empirical sample mean as the mean of the
# resampled tychenic means? 
estimateOffset = np.mean(tychenicMeans) - sampleMeans[whichMovie]

# How do the tychenic sample means distribute? How tight is this distribution? 
# Could the estimate reasonably have been +/- 0.1 from the real sample mean?
numBins = 101
plt.hist(tychenicMeans,numBins)
plt.xlabel('Tychenic sample means')
plt.ylabel('Count')

# Add the sample mean:
plt.plot([sampleMeans[whichMovie],sampleMeans[whichMovie]],[0,400],color='black',linewidth=0.5) 

#%% Let's add the CI

confidenceLevel = 95 # What confidence level (probability of containing 
# the empirical mean) is desired? Also try 99%
lowerBoundPercent = (100 - confidenceLevel)/2 # lower bound
upperBoundPercent = 100 - lowerBoundPercent # upper bound
lowerBoundIndex = round(numRepeats/100*lowerBoundPercent)-1 # what index?
upperBoundIndex = round(numRepeats/100*upperBoundPercent)-1 # what index?
sortedSamples = np.sort(tychenicMeans,axis=0)
lowerBound = sortedSamples[lowerBoundIndex] # What tychenic value consistutes the lower bound?
upperBound = sortedSamples[upperBoundIndex] # What tychenic value consistutes the upper bound?

# Add it to the plot:
numBins = 101
plt.hist(tychenicMeans,numBins)
plt.xlabel('Tychenic sample means')
plt.ylabel('Count')
plt.plot([sampleMeans[whichMovie],sampleMeans[whichMovie]],[0,400],color='black',linewidth=0.5)
plt.plot([lowerBound,lowerBound],[0,400],color='red',linewidth=0.5) 
plt.plot([upperBound,upperBound],[0,400],color='red',linewidth=0.5) 

#%% 2 Let's do a permutation test with this data, by drawing WITHOUT replacement

# Let's say we compare ratings of MATRIX II to ratings of MATRIX III
empiricalData1 = combinedData[:,1] # Ratings of M2
empiricalData2 = combinedData[:,2] # Ratings of M3
ourTestStat = np.mean(empiricalData1) - np.mean(empiricalData2)
# This is the creative hurdle. We just made this up on the stop
# Could have divided. Or taken the log. Or whatever else. Or taken the mean 
# of the differences instead of the differences of the mean. But let's do 
# this one, for now. We could have taken the ratio, which we will try next.

# What question should we always ask ourselves once we have a value? 
# "How likely is a test statistic like that just by chance?"
# --> Is it unlikely enough to make a decision - to be considered 
# "statistically significant" 
# Normally, for t or F or Chi-squared or known test statistics, we get this
# p value from a giant table at the back of the textbook. 
# But how does our test statistic distribute? 
# No idea. So let's create a "null distribution"

numReps = int(1e5) # This is how many times we'll draw WITHOUT replacement 
# to create the null distribution
jointData = np.concatenate((empiricalData1,empiricalData2)) # Stack them on 
# top of each other. For a joint representation.
n1 = len(empiricalData1) # How long one of them is
n2 = len(jointData) # Overall length
shuffledStats = np.empty([numReps,1]) # Initialize empty array
shuffledStats[:] = np.NaN # Then convert to NaN

# Joint Data has indices from 1 to 2xn1. 
# Step 1: Randomly shuffle those indices
# Step 2: Split them in the middle to create two artificial groups 
# Step 3: Repeat num reps times
# Step 4: Profit
for i in range(numReps):
    shuffledIndices = np.random.permutation(n2) # shuffle indices 0 to 2985
    shuffledGroup1 = jointData[shuffledIndices[:n1]]
    shuffledGroup2 = jointData[shuffledIndices[n1:]]
    shuffledStats[i,0] = np.mean(shuffledGroup1) - np.mean(shuffledGroup2)
    # Also try taking the ratio for line 148

#%% Now what? Let's plot the null distribution
plt.hist(shuffledStats,numBins)
plt.title('Null distribution vs. empirical value of our test statistic')
plt.plot([ourTestStat,ourTestStat],[0,4000],color='red',linewidth=1.0)

#%% Let's calculate the exact p-value

# Where is the shuffled stat greater than our test stat?:
temp1 = np.argwhere(shuffledStats > ourTestStat)

# How often is the test stat larger than the empirical one by chance?:
temp2 = len(temp1)

# Compute the p-value:
exactPvalue = temp2/len(shuffledStats)
print(exactPvalue)

# So how do you know when your num_rep is large enough?
# If the exact p-value doesn't change much if you run it again (stable - and
# far away from a decision boundary, like 0.05)

# Like here. It is <0.05. So it is significantly different.
# And the p is "exact". No assumptions about how the data is distributed,
# like in the parametric tests (assuming it is distributed normally). It is
# what it is - the sample.
