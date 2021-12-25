# Bayesian Statistics
# Code by Pascal Wallisch and Stephen Spivack
# Date: 12-25-21

#%% 1. The simple version of Bayes theorem
# This code calculates a posterior p(A|B) from a prior p(A), the likelihood of 
# B given A, and the baserate p(B). 
# p(A|B) = p(A) * p(B|A) / P(B)
prior = 0.05
likelihood = 1
baserate = 0.0625
posterior = prior*likelihood/baserate

# Suggestion for exploration: Try different numbers for prior, likelihood
# and baserate to see how it affects the posterior.

#%% 2. The explicit form of Bayes theorem
# This code implements the explicit version of Bayes theorem. 
# It calculates a posterior p(A|B) from a prior p(A), the likelihood of B
# given A (likelihood1) and likelihood of B given not A (likelihood2). 
# p(A|B) = p(A) * p(B|A) / ( p(A) * p(B|A) + (1-p(A)) * p(B|~A) )

prior = 0.3
likelihoodGivenA = 0.5
likelihoodGivenNotA = 0.25
posterior = (prior*likelihoodGivenA)/(prior*likelihoodGivenA+(1-prior)*likelihoodGivenNotA)

# Suggestion for exploration: Can you package segments 1) and 2) as a
# function - so you have a ready made Bayesian calculator?

#%% 3. The Bayes factor
pDH1 = 0.5 # probability of data given alternative hypothesis
pDH0 = 0.1 # probability of data given null hypothesis
bf10 = pDH1/pDH0 # The BF10 = 5 -> "Moderate evidence for H1"

# Suggestion for exploration: Try different likelihoods or framing it as BF01

#%% 4. Modeling the prior distribution, trying several alpha and beta parameters

# 0. Import packages:
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# 1. Init:
numParameters = 50 # We are going to loop through 50 values of alpha and beta
x = np.linspace(0,1,1001)

# 2. Run simulation and plot:
for ii in range(numParameters): # Loop through each alpha and beta parameter
    alpha = ii+1
    beta = ii+1
    prior = stats.beta.pdf(x,alpha,beta) # Modeling the prior as a beta distribution with parameters alpha = beta
    plt.plot(x,prior,color='red',linewidth=3) 
    ax = plt.gca() # get current axes
    ax.axes.yaxis.set_visible(False) # remove y-axis ticks and labels
    plt.ylim(0,8.5) # hardcoded ylim
    plt.xlabel(r'$\theta$') 
    plt.title('Prior distribution: Beta distribution with ' r'$\alpha$ =  {}  '.format(ii+1) + r'$\beta$ =  {}'.format(ii+1)) 
    plt.pause(0.1)

#%% 5. Now combining the prior distribution with the likelihood to yield the posterior distribution

# 1. Init:
numReps = 75 # how many sample sizes are we looping through?
# Each time we are going to increase n by 10!
x = np.linspace(0,1,1001)

# 2. Run simulation:
for ii in range(numReps): # Loop through each rep
    alpha = 30 # Clamping parameters at 30
    beta = 30
    # Modeling the prior as a beta distribution with parameters alpha = beta:
    prior = stats.beta.pdf(x,alpha,beta) 
    # Plot the data:
    plt.plot(x,prior,color='red',linewidth=3) 
    ax = plt.gca() 
    ax.axes.yaxis.set_visible(False) 
    plt.ylim(0,10) 
    plt.xlabel(r'$\theta$') 
    # Use a binomial distribution to represent the likelihood as a function of n:
    n = 10*ii + 10
    para = 0.35
    x2 = np.linspace(0,n,n+1)
    x2 = x2/n # Normalize to 1
    likelihood = stats.binom.pmf(np.linspace(0,n,n+1),n,para) * n/3
    plt.plot(np.linspace(0,1,len(likelihood)),likelihood,color='blue',linewidth=3) 
    # Calculate the posterior:
    posterior = stats.beta.pdf(x,para*n+alpha,n-para*n+beta)/3
    plt.plot(x,posterior,color='black',linewidth=3) 
    plt.title('Prior distribution: Beta distribution with ' r'$\alpha$ =  {}  '.format(alpha) + r'$\beta$ =  {}'.format(beta) + ' n =  {}'.format(n))
    plt.legend(['Prior', 'Likelihood', 'Posterior'])
    plt.pause(0.05)
    
#%% 6. The credible interval - Bayesian analogue to confidence intervals in frequentist statistics

totalProbability = sum(posterior)
cutoff = 50 #cutoff
binMiddle = 374 #Center 
fractionProbability = sum(posterior[binMiddle-cutoff:binMiddle+cutoff])/totalProbability
for qq in range(binMiddle-cutoff-1,binMiddle+cutoff):
    plt.plot([x[qq],x[qq]],[0,posterior[qq]],color='green',linewidth=2) 
alpha = 30
beta = 30
informativePrior = stats.beta.pdf(x,alpha,beta) 
plt.plot(x,informativePrior)
plt.xlabel(r'$\theta$') 
ax = plt.gca() 
ax.axes.yaxis.set_visible(False) 

#Here, we hard-coded everything. Never, ever do that in real life. It makes for code that is very hard to understand and maintain.
#Exercise for the reader: Replace the fixed "r" and "binmiddle" with the 95% center of sorted values, like in the tychenic resampling example in last lab.

#%% 7. Probabilistic modeling of finding drunks in bars (Philosophy: Coding saves thinking)

n = int(1e6) # Number of iterations
pDrinking = 1-0.9 # Probability of drinking = being in a bar
randomNumbers = np.random.uniform(0,1,n) # Draw random numbers from a uniform distribution from 0 to 1
barAttendance = np.zeros([n,3]) # Initial state of the bars
for ii in range(n):
    temp = randomNumbers[ii]
    if temp > pDrinking: # If he is drinking
        temp2 = np.random.randint(3) # draw random integer from 0 to 2
        if temp2 == 0: # Go to bar 1
            barAttendance[ii,0] = 1
        elif temp2 == 1: # Go to bar 2
            barAttendance[ii,1] = 1
        elif temp2 == 2: # Go to bar 3
            barAttendance[ii,2] = 1
            
#%% After running the code in the previous segment, we now have sata from a large 
# number of days, representing which bar they ended up in, if any.
# This allows us to implement the idea that the prior gets constantly updated with 
# new information, as we don't find the drunk in bars.

bar2Counter = 0
bar3Counter = 0
for ii in range(n): # Sequentially checking the bars, day by day and bar by bar. Keeping track of where the individual was located
    if int(sum(barAttendance[ii,0:2])) == 0: # If he is not in the first two bars (remember that the indexing range is *not* inclusive)
        bar2Counter = bar2Counter + 1 # Number of times this happened - they went to the first 2 bars and didn't find him there   
        bar3Counter = bar3Counter + barAttendance[ii,2]
probabilityBeingInThirdBarIfNotInFirstTwo = bar3Counter/bar2Counter
print(probabilityBeingInThirdBarIfNotInFirstTwo)