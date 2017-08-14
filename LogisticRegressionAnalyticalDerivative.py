import numpy as np 
import random

# Just wanted to see if I could model the function y = 2x1 - 3x2

def createDataset(numExamples, numDim):
	localX = np.random.randint(-10, high=10, size=(numExamples, numDim))
	localY = np.zeros([numExamples, 1])
	for i in range(numExamples):
		localY[i] = (2 * (localX[i,0])) - (3 * localX[i,1])
	return localX,localY

def getTrainBatch(examples, bSize, x, y):
	randomNum = random.randint(0,examples - bSize - 1)
	return x[randomNum:randomNum + bSize], y[randomNum:randomNum + bSize]

def computeGradient(labels, localZ, preds, inputs, numInBatch):
	dLdYPred = -(labels/preds) + (1 - labels)/(1 - preds)
	dYPreddZ = sigmoidDerivative(localZ)
	# Beause chain rule
	dLdZ = dLdYPred * dYPreddZ
	dZdW = inputs
	dLdW = np.dot(dLdZ.T, dZdW) / numInBatch
	return dLdW

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoidDerivative(x):
	return x * (1.0 - x)

# Hyperparameters
numPoints = 1000
numInputDim = 2
testSplit = 0.20
numTestExamples = int(numPoints * testSplit)
numTrainExamples = numPoints - numTestExamples 
numIterations = 1000
learningRate = 0.0001
batchSize = numTrainExamples

# Create dataset
X,Y = createDataset(numPoints, numInputDim)

# Split the dataset
xTrain, xTest = X[numTestExamples:], X[:numTestExamples]
yTrain, yTest = Y[numTestExamples:], Y[:numTestExamples]

# Create network 
W = np.random.rand(1, numInputDim)
b = np.random.rand(1, 1)

# Training
for i in range(numIterations):
	# Get training batch
	#xBatch, yBatch = getTrainBatch(numTrainExamples, batchSize, xTrain, yTrain)
	xBatch, yBatch = xTrain, yTrain
	# Forward pass
	z = np.matmul(xBatch, W.T) + b
	yPred = sigmoid(z[0])
	loss = -(yBatch * (np.log(yPred) + (1-yBatch) * np.log(1-yPred)))
	cost = np.sum(loss) / numTrainExamples
	gradient = computeGradient(yBatch, z, yPred, xBatch, numTrainExamples)
	# Weight update
	W = W - learningRate * gradient
	print 'Cost: ', cost