# New Statistics: Power and Effect Sizes
# Code by Pascal Wallisch and Stephen Spivack
# Date: 12-25-21

#%% 0. Initialize
import numpy as np
import numpy.matlib 
from scipy import stats
import matplotlib.pyplot as plt

#%% 1. Distributions of p values assuming true vs. false null hypothesis (H0)
# True null: try 0 for mean difference
# False null: try 0.25, 0.5 and 1 for mean difference

# Initialize variables:
sampleSize = 2000
numReps = int(1e4) # Number of repetitions in our simulation
meanDifference = 0 # Actual difference in sample means. Try 0 and other values.

# Draw from a random normal distribution with zero mean:
sata1 = np.random.normal(0,1,[sampleSize,numReps])

# Our 2nd sample. Same distribution, different mean:
sata2 = np.random.normal(0,1,[sampleSize,numReps]) + meanDifference

# Run a t-test, a lot of times:
t = np.empty([numReps,1]) # initialize empty array for t
t[:] = np.NaN # then convert to NaN
p = np.empty([numReps,1]) # initialize empty array for p
p[:] = np.NaN # then convert to NaN
for i in range(numReps): # loop through each rep
    t[i],p[i] = stats.ttest_ind(sata1[:,i],sata2[:,i]) # do the t-test
    
# Plot the data:
plt.hist(p,100)
plt.xlabel('p-value')
plt.ylabel('frequency')

print('Min p-value:',np.min(p))
print('Max p-value:',np.max(p))

#%% 2. Mean differences, effect sizes and significance
# Effect sizes as salvation?
# Central take home message: The same p value can correspond to dramatically
# different effect sizes.

# A. Initialize variables:
numReps = 500 # Number of repetitions in our simulation
sata = np.empty([numReps,3,2]) # Initialize empty 3D array for sata
sata[:] = np.NaN # then convert to NaN 
PvsE = np.empty([numReps,5]) # Initialize empty 2D array for PvsE
PvsE[:] = np.NaN # then convert to NaN

# B. Generate and analyze sata:
for i in range(numReps): # loop through each rep
    p = 0 # set p to 0
    while abs(p - 0.04) > 0.01: # Find datasets that are just about significant
        temp = np.random.normal(0,1,[3,2]) # Draw n = 3 (mu = 0, sigma = 1)
        t,p = stats.ttest_rel(temp[:,0],temp[:,1]) # paired t-test
    sata[i] = temp # store temp in sata array
    
    # sample size:
    PvsE[i,0] = len(sata[i,:,:]) # take the length of the z-stack dimension
    
    # significance level:
    t,p = stats.ttest_rel(sata[i,:,0],sata[i,:,1]) # paired t-test
    PvsE[i,1] = p 
    
    # effect size (computing cohen's d by hand):
    mean1 = np.mean(sata[i,:,0]) # mean of sample 1
    mean2 = np.mean(sata[i,:,1]) # mean of sample 2
    std1 = np.std(sata[i,:,0]) # std of sample 1
    std2 = np.std(sata[i,:,1]) # std of sample 2
    n1 = len(sata[i,:,0]) # size of sample 1
    n2 = len(sata[i,:,1]) # size of sample 2
    numerator = abs(mean1-mean2) # absolute value of mean difference
    denominator = np.sqrt((std1**2)/2 + (std2**2)/2) # pooled std
    d = numerator/denominator
    PvsE[i,2] = d
    
    # mean differences:
    PvsE[i,3] = abs(np.mean(sata[i,:,0]) - np.mean(sata[i,:,1]))
    
    # pooled standard deviation:
    PvsE[i,4] = np.sqrt((std1**2)/2 + (std2**2)/2)
    
# C. Plot it:
plt.subplot(2,3,1)
plt.hist(PvsE[:,1],20)
plt.title('p value')
plt.subplot(2,3,2)
plt.hist(PvsE[:,2],20)
plt.title('cohens d')
plt.subplot(2,3,3)
plt.hist(PvsE[:,3],20)
plt.title('mean diff')
plt.subplot(2,3,4)
plt.hist(PvsE[:,4],20)
plt.title('pooled sd')
plt.subplot(2,3,5)
plt.plot(PvsE[:,2],PvsE[:,3],'o',markersize=.5)
plt.xlabel('cohens d')
plt.ylabel('abs mean diff')
plt.subplot(2,3,6)
plt.plot(PvsE[:,2],PvsE[:,4],'o',markersize=.5)
plt.xlabel('cohens d')
plt.ylabel('pooled sd')

#%% 3. PPV - "positive predictive value" - what we actually want to know.
# The *post* study probability that a significant result is actually true.
# The probability that something that is significant is actually true.

# Initialize variables:
alpha = 0.05 # fisher's choice
beta = 0.2 # classic choice
R = 0.5 # we want to know. this is our prior belief (Ratio of true to false effects in a field)
ppv = ((1-beta)*R)/(R-beta*R+alpha) 

# What if we are agnostic - let's explore all Rs:
R = np.linspace(0,1,101) # 0 to 1 in .01 increments (ratio of true to false effects)
ppv = np.empty([len(R),1]) # initialize empty array
ppv[:] = np.NaN # convert to NaN
for i in range(len(R)): # loop through each R
    ppv[i] = ((1-beta)*R[i])/(R[i]-beta*R[i]+alpha)
    
# Plot it:
plt.plot(R,ppv)
plt.xlabel('R')
plt.ylabel('ppv')

#%% So far, we power-clamped at 0.8 - what if we vary power too?

beta = np.linspace(1,0,101) # power = 1 - beta
R = np.linspace(0,1,101) 
ppv = np.empty([len(R),len(beta)]) # initialize empty array
ppv[:] = np.NaN # convert to NaN
for i in range(len(R)): # loop through each R
    for j in range(len(beta)): # loop through each beta
        ppv[i,j] = ((1-beta[j])*R[i])/(R[i]-beta[j]*R[i]+alpha)
        
# Summarizing the Iannidis paper in one figure:
x = R # 1d array
y = beta # 1d array
x, y = np.meshgrid(x,y) # make a meshgrid out of x and y
z = ppv # 2d array
fig = plt.figure() # init figure
ax = fig.gca(projection='3d') # project into 3d space
surf = ax.plot_surface(x,y,z) # make surface plot
ax.set_xlabel('R') # add xlabel 
ax.set_ylabel('beta') # add ylabel 
ax.set_zlabel('ppv') # add zlabel 

# To see: Tradeoff between alpha, beta, and R.

#%% 4. Flexible stopping - the p-value *always* gets below alpha, eventually. Just by flexible stopping alone.
#This is the weaponization of sampling error for bad ends - to pursue careerist goals that hurt society

# Init:
n = 5 # Starting n - here, 5
p = 1 # Starting p - value - let's say 1
alpha = 0.05 # What is our alpha level?
droppingToo = 0 # If this flag is on, we don't just flexibly stop, we also drop people, if they increase the p-value
nCont = np.array([]) # Initialize the container that will hold our ns 
pCont = np.array([]) # And one for the corresponding p-values
data = np.random.normal(0,1,[n,2]) # %Randomly draw data from n people from a normal distribution, in 
# both conditions, like for an A/B test. Note that there is no effect here. Just randomness. 

# Run simulation:
while p > alpha: # As long as the p-value is not significant yet
    t,p = stats.ttest_ind(data[:,0],data[:,1]) # Do a t-test on the data, columns = A vs. B
    nCont = np.append(nCont,n) # Capture the n for this test 
    pCont = np.append(pCont,p) # And the corresponding p-value
    tempP = 1  # Initialize a new p-value
    if droppingToo == 0: # Flexible stopping only
        data = np.concatenate((data,np.random.normal(0,1,[1,2])),axis=0) # Then add data from 1 (one) new participant to the dataset, one per condition
    elif droppingToo == 1:
        while tempP > p: # While our new p is larger than our old p
            tempData = np.concatenate((data,np.random.normal(0,1,[1,2])),axis=0) # Try new people until they lower our p-value. Obviously, there is something wrong with those who don't help our p-value
            t,tempP = stats.ttest_ind(tempData[:,0],tempData[:,1]) # Do a t-test on the stop with the provisional new data, if they make our p-value worse, we don't even enter them into the dataset
            break
        data = np.concatenate((data,tempData),axis=0) # Add them to the dataset if (and only if) they lower the p-value
    n = n + 1 # Update the n accordingly (we could simply get it from length(n), but whatever)

# Plot the data:
plt.plot(nCont,pCont,color='red',linewidth=0.5)
plt.plot(nCont,pCont,'o',color='black',markersize=2)
plt.xlabel('Participants')
plt.ylabel('p')
plt.xlim(5,n)
plt.plot([5,n],[alpha,alpha],color='black',linewidth=0.5,linestyle='dashed') # draw decision threshold as a line
plt.title('Final p-value = {:.3f}'.format(p))

#%% 5. Funnel plots - these are a meta-analytic tool. 
# Plotting effect size vs. power - probability of a hypothesis test of finding 
# an effect if there is an effect to be found

# As you increase power, the effects cluster around the "real" effect
# If you run low-powered stuff, you will be at the bottom of the funnel and
# your effect sizes will jump all over the place. 

# Initialize variables:
sampleSize = np.linspace(5,250,246) # We vary sample size from 5 to 250
effectSize = 0 # The real effect size
repeats = int(1e2) # In reality, you would do many more than that
meanDifference = np.zeros([repeats,len(sampleSize)]) # preallocate

# Calculations:
for r in range(repeats): # loop through each repeat
    for i in range(len(sampleSize)): # loop through each sample size
        tempSample = int(sampleSize[i]) # what is our n this time?
        temp = np.random.normal(0,1,tempSample) + effectSize
        temp2 = np.random.normal(0,1,tempSample)
        meanDifference[r,i] = np.mean(temp) - np.mean(temp2)
        
# Learning effect: Larger n will converge to real effect
# Lower n allows fishing for noise. Instead of doing 1 high powered
# experiment, n = 250, do 10 n = 25 experiments and publish the one
# that becomes significant. Brutal.

# To save time, let's linearize that:
linearizedMeanDiff = np.ndarray.flatten(meanDifference) 
temp = np.matlib.repmat(sampleSize,repeats,1) 
linearizedSample = np.ndarray.flatten(temp)

# Now we can do the funnel plot:
plt.plot(linearizedMeanDiff,linearizedSample,'o',markersize=.5)
plt.xlabel('Observed effect size')
plt.ylabel('Sample size')

#%% 6. Powerscape
# See: https://blog.pascallisch.net/brighter-than-the-sun-introducing-powerscape/

# Initialize variables:
popSize = int(1e3) # Size of the population
nnMax = 250 # Maximal sample size to be considered
nnMin = 5 # Minimal sample size to be considered
sampleSize = np.linspace(nnMin,nnMax,246) # array of each sample size
effectSize = np.linspace(0,1.5,31) # From 0 to 1.5, in .05 increments
# As std is one, effectSize will be in units of std
pwr = np.empty([len(sampleSize),len(effectSize)]) # initialize power array
pwr[:] = np.NaN # then conver to nan

# Run calculations:
for es in range(len(effectSize)): # loop through each effect size (31 total)
    A = np.random.normal(0,1,[popSize,2]) # Get the population of random 
    # numbers for each effect size - 2 columns, 1000 rows
    A[:,1] = A[:,1] + effectSize[es] # Add effect size to 2nd column/set
    for n in range(len(sampleSize)): # loop through each sample size
        mm = int(2e1) # Number of repeats
        significances = np.empty([mm,1]) # preallocate
        significances[:] = np.NaN
        for i in range(mm): # Do this mm times for each sample size
            sampInd = np.random.randint(0,popSize,[n+5,2]) # subsample
            # we add 5 to n because n is indexed at 0 but our min n is 5
            drawnSample = np.empty([n+5,2]) # initialize empty
            # drawn_sample starts as 5x2 and with each iteration adds one row
            drawnSample[:] = np.NaN # convert to NaN
            drawnSample[:,0] = A[sampInd[:,0],0] 
            drawnSample[:,1] = A[sampInd[:,0],1]
            t,p = stats.ttest_ind(drawnSample[:,0],drawnSample[:,1])
            if p < .05: # assuming our alpha is 0.05
                significances[i,0] = 1 # if significant, assign 1
            else:
                significances[i,0] = 0 # if ~significant, assign 0
        pwr[n,es] = sum(significances)/mm*100 # compute power
        
# Plot it:
plt.pcolor(pwr) #create a pseudocolor plot with a non-regular rectangular grid
# color = perfect significant effects
plt.xlabel('real effect size (mean diff in SD)')
plt.ylabel('sample size (n)')
plt.title('powerscape t-test') # color represents proportion significant effects
plt.colorbar()