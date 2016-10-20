import numpy as np 
import math

Xtrain = [1,2,3,4] #Stores the training inputs
Ytrain = [3,5,7,9] #Stores the training labels

#Initialization of weights with random numbers
theta0 = np.random.uniform(0,1)
theta1 = np.random.uniform(0,1)
learningRate = .1
numEpochs =10

def hypothesis(x):
	return (theta0 + theta1*x)

def costFunction():
	for i, j in zip(range(Xtrain),range(Ytrain)):
		temp = math.pow((hypothesis(i) - j),2)

def weightUpdate():
	# Working on how to represent taking the derivative of the
	# cost function with represent to the theta values
	theta0 = theta0 - learningRate*()
	theta1 = theta1 - learningRate*()

Xtest = 5
for i in range(1,numEpochs):
	print ('')
	#weightUpdate()
print (hypothesis(Xtest))
