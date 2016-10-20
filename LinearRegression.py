import numpy as np 
import math

Xtrain = [1,2,3,4] #Stores the training inputs
Ytrain = [3,5,7,9] #Stores the training labels

#Initialization of weights with random numbers
learningRate = .001
numEpochs =100

def costFunction(t0, t1):
	loss = 0
	for i, j in zip(Xtrain,Ytrain):
		temp = math.pow(((t0 + t1*i) - j),2)
		loss += temp
	return loss

def weightUpdate(withRespectTo, t0, t1):
	if (withRespectTo == "theta0"):
		t0 = t0 - learningRate*(derivative(withRespectTo, t0, t1))
		return t0
	else: #has to be wrt to theta1
		t1 = t1 - learningRate*(derivative(withRespectTo, t0, t1))
		return t1
	
def derivative(withRespectTo, t0, t1):
	h = 1./1000.
	if (withRespectTo == "theta0"):
		rise = costFunction(t0 + h, t1) - costFunction(t0,t1)
	else: #has to be wrt to theta1
		rise = costFunction(t0 , t1 + h) - costFunction(t0,t1)
	run = h
	slope = rise/run
	return slope

theta0 = np.random.uniform(0,1)
theta1 = np.random.uniform(0,1)
Xtest = 5
for i in range(1,numEpochs):
	weightUpdate("theta0", theta0, theta1)
	weightUpdate("theta1", theta0, theta1)
	print (theta1)
print (theta0 + theta1*Xtest)
