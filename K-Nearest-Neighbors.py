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
Xtrain = [[3,1],[3,2],[1,2],[1,3]]
Ytrain = [1,1,0,0]
#Hyperparamters
numTrainExamples = len(Xtrain)
numEpochs = 200
numNeighbors = 3

#TRYING TO FIGURE OUT SORTING LISTS WITHIN LISTS

l = [[3,2,10],[4,3,19],[2,2,12]]
print l
sorted(l, key=itemgetter(0))
print l

Xtest = [3,1.2]
minDistance = sys.maxint
index = 0

#STILL WORKING ON IT

distanceAndLocation = []
for x in range(0,numTrainExamples):
	distance = distanceBetween(Xtrain[x],Xtest)
	distanceAndLocation.append(distance)
	



