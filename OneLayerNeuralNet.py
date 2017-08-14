import numpy as np 
import random

# One layer neural network in Numpy

# Just wanted to see if I could model the function y = 0.5x1^2 - 2x2

def createDataset(numExamples, numDim):
	localX = np.random.randint(-10, high=10, size=(numExamples, numDim))
	localY = np.zeros([numExamples, 1])
	for i in range(numExamples):
		localY[i] = (0.5 * (localX[i,0])) - (2 * localX[i,1])
	return localX,localY

def getTrainBatch(examples, bSize, x, y):
	randomNum = random.randint(0,examples - bSize - 1)
	return x[randomNum:randomNum + bSize], y[randomNum:randomNum + bSize]

def computeGradient(preds, inputs, loss):
	dLdW = np.matmul(inputs.T, loss * reluDerivative(preds))
	return dLdW

def relu(x):
	return np.maximum(x, 0, x)

def reluDerivative(x):
	derivative = x
	derivative[derivative <= 0] = 0
	derivative[derivative > 0] = 1
	return derivative

# Hyperparameters
numPoints = 1000
numInputDim = 2
testSplit = 0.20
numTestExamples = int(numPoints * testSplit)
numTrainExamples = numPoints - numTestExamples 
batchSize = 1
numIterations = 10000
learningRate = 0.0001

# Create dataset
X,Y = createDataset(numPoints, numInputDim)

# Split the dataset
xTrain, xTest = X[numTestExamples:], X[:numTestExamples]
yTrain, yTest = Y[numTestExamples:], Y[:numTestExamples]

# Create network 
W = np.random.rand(numInputDim, 1)
b = np.random.rand(batchSize, 1)

# Training
for i in range(numIterations):
	# Get training batch
	xBatch, yBatch = getTrainBatch(numTrainExamples, batchSize, xTrain, yTrain)
	# Forward pass
	yNoActivation = np.matmul(xBatch, W) + b
	yPred = relu(yNoActivation)
	# Loss 
	print yPred
	print yBatch
	loss = np.sum(np.square(yPred - yBatch))
	# Backward pass - Compute dL/dW
	gradient = computeGradient(yNoActivation, xBatch, loss)
	# Weight update
	W = W - learningRate * gradient
	print learningRate * gradient
	print ('Loss at iteration %d is %f', i, loss)

