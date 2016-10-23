import numpy as np 
import math

#This program is just a rough reimplementation of multivariate linear regression in Python. It's not particularly
#optimized in any way but it does give a sense of backpropagation, computing the loss function, and
#updating the weights. The derivatives are taken numerically, instead of analytically. Numeric
#derivatives are a bit slower and less intuitive, but do still work in this case.

Xtrain = [[3,1],[3,2],[1,2],[1,3]] #Stores the training inputs
Ytrain = [3,5,7,9] #Stores the training labels

#Hyperparameters 
learningRate = .01
numEpochs =1000

#In this case, our hypothesis is in the form of a model representing multivariate
#linear regression. Theta is a 1 x N vector where N  is the number of weights
def hypothesis(T,x):
	return (T[0] + T*x)

#Our loss function is the classic mean squared error form
def costFunction(T):
	loss = 0
	for i, j in zip(Xtrain,Ytrain):
		temp = math.pow((hypothesis(t0,t1,i) - j),2)
		loss += temp
	return loss

#Weight updates are done by taking the derivative of the loss function 
#with respect to each of the theta values. 
def weightUpdate(withRespectTo, t0, t1):
	if (withRespectTo == "theta0"):
		t0 = t0 - learningRate*(derivative(withRespectTo, t0, t1))
		return t0
	else: #has to be wrt to theta1
		t1 = t1 - learningRate*(derivative(withRespectTo, t0, t1))
		return t1
	
#Evaluating a numerical deerivative
def derivative(withRespectTo, t0, t1):
	h = 1./1000.
	if (withRespectTo == "theta0"):
		rise = costFunction(t0 + h, t1) - costFunction(t0,t1)
	else: #has to be wrt to theta1
		rise = costFunction(t0 , t1 + h) - costFunction(t0,t1)
	run = h
	slope = rise/run
	return slope

#Random initialization of the theta values
theta0 = np.random.uniform(0,1)
theta1 = np.random.uniform(0,1)
#Test value
Xtest = 5
for i in range(0,numEpochs):
	theta0 = weightUpdate("theta0", theta0, theta1)
	theta1 = weightUpdate("theta1", theta0, theta1)
print (hypothesis(theta0,theta1,Xtest))
