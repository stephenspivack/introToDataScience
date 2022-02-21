# Understanding least squares from a linear algebra perspective
# Code by: Pascal Wallisch and Stephen Spivack
# Date: 02-21-22

#%% Recall the principles of vector projection from the lecture:
    
import numpy as np
import matplotlib.pyplot as plt

vec1 = np.array([0,0.5]) #Vector 1
vec2 = np.array([1,1]) #Vector 2
magVec1 = np.sqrt(vec1[0]**2 + vec1[1]**2) #Magnitude of vector 1
magVec2 = np.sqrt(vec2[0]**2 + vec2[1]**2) #Magnitude of vector 2 
dotProduct = np.dot(vec1,vec2) #Using a function to get the dot product 
angleBetween = np.degrees(np.arccos(dotProduct/(magVec1*magVec2))) #What is the angle between the vectors?
uVec = vec2/magVec2 # Creating a unit vector out of vec2 by dividing by magnitude
p = magVec1 * np.cos(np.deg2rad(angleBetween)) # The projection direction
projVec = p * uVec # That's the actual projected vector, yielded by p multiplied with the unit vector

plt.plot([0,vec1[0]],[0,vec1[1]],color='purple',linewidth=2) # Plot vec1 in purple
plt.plot([0,uVec[0]],[0,uVec[1]],color='blue',linewidth=2) # Plot uVec in blue
plt.plot([0,projVec[0]],[0,projVec[1]],color='red',linewidth=2) # Plot the projection of vec1 onto vec2 in red
plt.axis('equal') #Make sure aspect ratio is the same

#%% Now Say we have an experiment with 3 trials - which is nice because two would
# be trivial and we can still visualize 3 dimensions
# We want to know how "performance" is related to lighting (luminance) in a room

xLuminance = np.array([1,2,3]) # A column vector of IV in cd/m^2 (Candela per square meter)
yPerformance = 1.5 * xLuminance + 0.5 + np.random.normal(0,1,len(xLuminance)) * 2

# Make a nice plot of the situation:
plt.plot(xLuminance,yPerformance,'o',markersize=10)
plt.xlabel('Luminance in cd/m^2')
plt.ylabel('Performance in arbitrary units')

# This is legit, but to see how the equation (which works) works
# let's look at this graphically. Or geometrically.
# To do that, we need to express the experiment visually.
# Now, each trial is a dimension.
# Now what? How do we find the solution?
# Plot the entire experiment of inputs and outputs as vectors 

fig = plt.figure() # init figure
ax = fig.gca(projection='3d') # project into 3d space
ax.plot3D([0,xLuminance[0]],[0,xLuminance[1]],[0,xLuminance[2]],color='blue',linewidth=2) 
ax.plot3D([0,yPerformance[0]],[0,yPerformance[1]],[0,yPerformance[2]],color='green',linewidth=2) 
plt.legend(['Luminance','Performance']) 
ax.set_xlabel('Trial 1') 
ax.set_ylabel('Trial 2') 
ax.set_zlabel('Trial 3') 

#%% Now, let's actually use the formula we derived
# We use the projection formula to find beta to minimize the distance
# between beta*input and output. Output = beta*input + error

beta = np.dot(yPerformance,xLuminance)/np.dot(xLuminance,xLuminance) # Find the beta
prediction = beta * xLuminance # Make a prediction (simplest possible)

# Add this to the plot - the plot thickens:
fig = plt.figure() # init figure
ax = fig.gca(projection='3d') # project into 3d space
ax.plot3D([0,xLuminance[0]],[0,xLuminance[1]],[0,xLuminance[2]],color='blue',linewidth=2) 
ax.plot3D([0,yPerformance[0]],[0,yPerformance[1]],[0,yPerformance[2]],color='green',linewidth=2) 
ax.plot3D([0,prediction[0]],[0,prediction[1]],[0,prediction[2]],color='black',linewidth=3,linestyle='dotted') 
plt.legend(['Luminance','Performance','Prediction']) 
ax.set_xlabel('Trial 1') 
ax.set_ylabel('Trial 2') 
ax.set_zlabel('Trial 3') 

#%% Let's explicitly add the distance between the two (prediction and outcome)

fig = plt.figure() # init figure
ax = fig.gca(projection='3d') # project into 3d space
ax.plot3D([0,xLuminance[0]],[0,xLuminance[1]],[0,xLuminance[2]],color='blue',linewidth=2) 
ax.plot3D([0,yPerformance[0]],[0,yPerformance[1]],[0,yPerformance[2]],color='green',linewidth=2) 
ax.plot3D([0,prediction[0]],[0,prediction[1]],[0,prediction[2]],color='black',linewidth=3,linestyle='dotted')
ax.plot3D([yPerformance[0],prediction[0]],[yPerformance[1],prediction[1]],[yPerformance[2],prediction[2]],color='red',linewidth=2)  
plt.legend(['Luminance','Performance','Prediction','Error']) 
ax.set_xlabel('Trial 1') 
ax.set_ylabel('Trial 2') 
ax.set_zlabel('Trial 3') 

#%% Now that we convinced ourselves that this is in fact the correct beta (geometrically)
# we can go back and plot the solution
# We could open the old figure again, but let's start from scratch
# What if we had 20 measurements (20 trials)?
maxLuminance = 10
xLuminance = np.linspace(0.5,maxLuminance,20) # A column vector of IV in cd/m^2 - 20 luminance values
yPerformance = 1.5 * xLuminance + 0.5 + np.random.normal(0,1,len(xLuminance)) * 2 # Noisy integrate and fire
beta = np.dot(yPerformance,xLuminance)/np.dot(xLuminance,xLuminance) # Find the beta
prediction = beta * xLuminance # Make a prediction (simplest possible)
regressionLineX = np.linspace(0,maxLuminance,10) # Gives us 10 equally spaced numbers between 0 and 10. Intrapolation, x-base
regressionLineY = beta * regressionLineX # Find the ys of the regression line
plt.plot(xLuminance,yPerformance,'o',markersize=5) # Plot the data
plt.plot(regressionLineX,regressionLineY,color='black') # Plot regression line
plt.plot([xLuminance,xLuminance],[prediction,yPerformance],color='red') # Residuals

#%% Multiple regression - simplest case: Adding a constant to the regression equation
# Even having more than one predictor makes it a multiple regression, even
# if it is a constant.

# We need to represent the baseline
# We need as many baselines as there are trials, and it always should have
# the same value. Because we assume this is a constant

baseline = np.ones(len(xLuminance))
designMatrix = np.column_stack([xLuminance,baseline])
betas = np.dot(np.linalg.inv(np.dot(np.transpose(designMatrix),designMatrix)),np.dot(np.transpose(designMatrix),yPerformance.reshape(len(yPerformance),1)))

#%% Let's plot the new regression line on top of the old figure

regressionLineYBar = betas[0]*regressionLineX + betas[1]*np.ones(len(regressionLineX))
plt.plot(regressionLineX,regressionLineYBar,color='magenta') # New regression line
plt.plot(xLuminance,yPerformance,'o',markersize=5) 
plt.plot(regressionLineX,regressionLineY,color='black') 
plt.plot([xLuminance,xLuminance],[prediction,yPerformance],color='red')

#%% Explaining how beta is determined by "dropping marbles" - which beta is just right?

# Let's explore the space of betas
startExploration = beta - 2
endExploration = beta + 2
numBeta = 200
testBetas = np.linspace(startExploration,endExploration,numBeta) # 200 betas 
# between the actual beta and +/- 2. Let's just go through them and try them

distanceSum = np.empty([numBeta,1]) # Init container
distanceSum[:] = np.NaN # Convert to NaN
for ii in range(numBeta):
    prediction = testBetas[ii] * xLuminance # Do this numBeta times
    # We now need to introduce a distance metric
    # We start with sum of squares (the most commonly used)
    distanceSum[ii] = sum((prediction-yPerformance)**2) # Sum of squares
    # distanceSum[ii] = sum(prediction-yPerformance) # Simple summed deviations
    # distanceSum[ii] = sum(np.log(prediction-yPerformance)) # Absolute summed deviations
    
# Let's plot that  
plt.plot(testBetas,distanceSum,color='blue',linewidth=3)
# We also want to indicate with a line where the original beta was
plt.plot([beta,beta],[0,3000],color='magenta')
plt.xlabel('Beta')
plt.ylabel('Sum of squares')

#%% Going deeper into optimization - trying different metrics - what 
# is special about the sum of squared differences?

startExploration = beta - 2
endExploration = beta + 2
numBeta = 200
testBetas = np.linspace(startExploration,endExploration,numBeta)
distanceSum = np.empty([numBeta,4]) # Init container
distanceSum[:] = np.NaN # Convert to NaN

for ii in range(numBeta):
    prediction = testBetas[ii] * xLuminance
    distanceSum[ii,0] = sum(prediction-yPerformance) # Simple
    distanceSum[ii,1] = sum((prediction-yPerformance)**2) # Sum of squares
    distanceSum[ii,2] = sum(abs(prediction-yPerformance)) # Absolute value
    distanceSum[ii,3] = sum(np.log(1 + (prediction-yPerformance)**2)) # Lorentzian

for ii in range(int(np.size(distanceSum)/len(distanceSum))):
    plt.subplot(2,2,ii+1)
    plt.plot(testBetas,distanceSum[:,ii])
    if ii == 0:
        plt.title('Summed differences')
    elif ii == 1:
        plt.title('Sum of square differences')
    elif ii == 2:
        plt.title('Sum of absolute differences')
    else:
        plt.title('Lorentzian') 
        
#%% Sweeping the line around, as seen in lecture

for ii in range(numBeta):
    fig = plt.figure() #Open figure, call it fig
    plt.subplot(1,2,1)
    prediction = testBetas[ii]*xLuminance
    plt.plot(xLuminance,yPerformance,'o',markersize=5,color='blue') 
    regressionLineX = np.linspace(0,10,100)
    regressionLineY = testBetas[ii] * regressionLineX
    plt.plot(regressionLineX,regressionLineY,color='black')
    plt.plot([xLuminance,xLuminance],[prediction,yPerformance],linewidth=0.5,color='red')
    plt.xlim(0,10)
    plt.ylim(-5,25)
    plt.xlabel('Luminance in cd/m^2')
    plt.ylabel('Performance (arbitrary units)')
    
    plt.subplot(1,2,2)
    plt.plot(testBetas[0:ii],distanceSum[0:ii,1],'o',markersize=0.5,color='black') 
    plt.xlim(startExploration,endExploration)
    plt.ylim(0,max(distanceSum[:,1]))
    plt.xlabel('Beta')
    plt.ylabel('Distance')
    plt.pause(0.005)
