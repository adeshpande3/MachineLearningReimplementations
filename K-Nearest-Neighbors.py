from __future__ import division
import numpy as np 
import math
from operator import itemgetter
import sys

#This program is just a rough reimplementation of k nearest neighbors in Python. It's not particularly
#optimized in any way but it does give a sense of how the the algorithm chooses the closest
#neighbors to a particular test vector, and then how the output class is determined. 

#Distance between 2 points
def distanceBetween(point1,point2):
	return math.pow((math.pow(point1[0] - point2[0],2) + math.pow(point1[1] - point2[1],2)),0.5)

#Visualize these data points on an 4x4 xy graph
Xtrain = [[3,1],[3,2],[1,2],[1,3],[4,4],[5,5],[5,7],[7,5],[8,8]]
Ytrain = [2,1,2,1,1,0,0,0,1]
#Hyperparamters
numTrainExamples = len(Xtrain)
numNeighbors = 5
numClasses = 3 #Classes have to be labeled starting from 0...numCLasses - 1

Xtest = [2,1.2]
minDistance = sys.maxint

distanceAndLocation = []
for x in range(0,numTrainExamples):
	distance = distanceBetween(Xtrain[x],Xtest)
	distanceAndLocation.append([Xtrain[x], distance, Ytrain[x]])
distanceAndLocation = sorted(distanceAndLocation, key=itemgetter(1))	

if len(distanceAndLocation) >= numNeighbors:
	classCount = np.zeros(numClasses)
	for i in range(0,numNeighbors):
		temp = distanceAndLocation[i]
		classCount[temp[2]] = classCount[temp[2]] + 1
	maxCount = 0
	index = 0
	for i in range(0,len(classCount)):
		if (classCount[i] > maxCount):
			maxCount = classCount[i]
			index = i
	print classCount
	print index
else:
	print 'Number of points less than number of neighbors'





