import numpy as np 
import math

#This program is just a rough reimplementation of logistic regression in Python. It's not particularly
#optimized in any way but it does give a sense of backpropagation, computing the loss function, and
#updating the weights. The derivatives are taken numerically, instead of analytically. Numeric
#derivatives are a bit slower and less intuitive, but do still work in this case.

Xtrain = [5,6,10,7,4]
Ytrain = [1,1,0,0,1] #Stores the training labels

#Hyperparameters 
numTrainingExamples = 5
learningRate = .1
numEpochs =1000

#In this case, our hypothesis is in the form of a model representing logistic
#regression
def hypothesis(t0,t1,x):
	return (1/(1+np.exp(-(t0 + t1*x))))

#Our loss function is different depending on whether the y value is 0 or 1
def costFunction(t0, t1):
	loss = 0
	for i, j in zip(Xtrain,Ytrain):
		temp = (-j*math.log(hypothesis(t0,t1,i))) - (1-j)*math.log(1 - hypothesis(t0,t1,i))
		loss += temp
	return loss/numTrainingExamples

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
