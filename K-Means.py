from __future__ import division
import numpy as np 
import math

#This program is just a rough reimplementation of k means in Python. It's not particularly
#optimized in any way but it does give a sense of how the clusters are initially created
#and how the centroids are moved around with different training data. 

#Distance between 2 points
def distanceBetween(point1,point2):
	return math.pow((math.pow(point1[0] - point2[0],2) + math.pow(point1[1] - point2[1],2)),0.5)

#Visualize these data points on an 4x4 xy graph
Xtrain = [[3,1],[3,2],[1,2],[1,3]]
#Hyperparamters
numTrainExamples = len(Xtrain)
numEpochs = 200
numClusters = 2

#Creates two randomly positioned cluster centroids
clusterCentroidOne = [np.random.uniform(0,4),np.random.uniform(0,4)]
clusterCentroidTwo = [np.random.uniform(0,4),np.random.uniform(0,4)]

#Repeats process for a number of iterations
for i in range(0,numEpochs):
	clusterOne = []
	clusterTwo = []
	for x in Xtrain:
		distanceToClusterOne = distanceBetween(clusterCentroidOne,x)
		distanceToClusterTwo = distanceBetween(clusterCentroidTwo,x)
		if (distanceToClusterOne < distanceToClusterTwo):
			clusterOne.append(x)
		else:
			clusterTwo.append(x)
		xTotal = 0
		yTotal = 0
		for y in clusterOne:
			xTotal += y[0]
			yTotal += y[1]
		if len(clusterOne) != 0:
			clusterCentroidOne[0] = xTotal/len(clusterOne)
			clusterCentroidOne[1] = yTotal/len(clusterOne)

		xTotal = 0
		yTotal = 0
		for y in clusterTwo:
			xTotal += y[0]
			yTotal += y[1]
		if len(clusterTwo) != 0:
			clusterCentroidTwo[0] = xTotal/len(clusterTwo)
			clusterCentroidTwo[1] = yTotal/len(clusterTwo)

print clusterCentroidTwo
print clusterCentroidOne


