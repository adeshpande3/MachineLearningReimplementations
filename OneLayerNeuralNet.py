import numpy as np 
import random

# One layer neural network in Numpy

# Just wanted to see if I could model the function y = 0.5x1^2 - 2x2

def createDataset(numExamples, numDim):
	localX = np.random.randint(-100, high=100, size=(numExamples, numDim))
	localY = np.zeros([numExamples, 1])
	for i in range(numExamples):
		localY[i] = (0.5 * np.square(localX[i,0])) - (2 * localX[i,1])
	return localX,localY

def getTrainBatch(examples, bSize, x, y):
	randomNum = random.randint(0,examples - bSize - 1)
	return x[randomNum:randomNum + bSize], y[randomNum:randomNum + bSize]

def computeGradient(weights, preds, labels, inputs):
	gradientMatrix = np.zeros_like(weights)
	# TODO This isn't right
	dLdW = 2 * (preds - labels) * inputs
	return dLdW

def relu(x):
	return np.maximum(x, 0, x)

# Hyperparameters
numTrainExamples = 1000
numInputDim = 2
testSplit = 0.20
numTestExamples = int(numTrainExamples * testSplit)
batchSize = 24
numIterations = 1000
learningRate = 0.01

# Create dataset
X,Y = createDataset(numTrainExamples, numInputDim)

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
	yPred = relu(np.matmul(xBatch, W) + b)
	# Loss 
	mseLoss = np.sum(np.square(yPred - yBatch))
	# Backward pass - Compute dL/dW
	gradient = computeGradient(W, yPred, yBatch, xBatch)
	# Weight update
	W = W - learningRate * gradient
	print ('Loss at iteration %d is %f', i, mseLoss)

