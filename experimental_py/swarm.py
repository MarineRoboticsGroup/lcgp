import time
import numpy as np

import graph
import math_utils

class Swarm:

	def __init__(self, sensingRadius):
		self.sensingRadius = sensingRadius
		self.robot_graph = graph.Graph()

	####### Swarm Utils #######

	def initializeSwarm(self, bounds, formation='square', nRobots=None, minEigval=0.75):
		# intialize formation and edges
		self.startConfig = formation
		self.minEigval = minEigval
		self.robot_graph.removeAllNodes()
		self.robot_graph.removeAllEdges()

		if formation.lower() == 'square':
			self.robot_graph.initializeSquare()
		elif formation.lower() == 'test6':
			self.robot_graph.initializeTest6()
		elif formation.lower() == 'test8':
			self.robot_graph.initializeTest8()
		elif formation.lower() == 'random':
			self.robot_graph.initializeRandomConfig(nRobots, bounds)
		else:
			print("The given formation is not valid\n")
			raise NotImplementedError

		print("Reordering Robots as Needed")
		# self.reorderRobotsBasedOnConnectivity()


		self.updateSwarm()
		self.relDistanceMatrix = self.robot_graph.getStiffnessMatrix()
	
	def initializeSwarmFromLocationListTuples(self, locList):
		# intialize formation and edges
		self.robot_graph.removeAllNodes()
		self.robot_graph.removeAllEdges()

		for loc in locList:
			self.robot_graph.addNode(loc[0], loc[1])

		self.updateSwarm()
		self.relDistanceMatrix = self.robot_graph.getStiffnessMatrix()

	def initializeSwarmFromLocationList(self, locList):
		# intialize formation and edges
		assert(len(locList)%2 == 0)
		self.robot_graph.removeAllNodes()
		self.robot_graph.removeAllEdges()

		for i in range(int(len(locList)/2)):
			self.robot_graph.addNode(locList[2*i], locList[2*i+1])

		self.updateSwarm()
		self.relDistanceMatrix = self.robot_graph.getStiffnessMatrix()

	def reorderRobotsBasedOnConnectivity(self):
		raise NotImplementedError

	def updateSwarm(self):
		startPlanning = time.time()
		self.robot_graph.updateEdgesByRadius(self.sensingRadius)
		endPlanning= time.time()

		startPlanning = time.time()
		self.relDistanceMatrix = self.robot_graph.getStiffnessMatrix()
		endPlanning= time.time()
		
		# print('Time Updating:', endPlanning - startPlanning)
		# print('Time Building Matrix:', endPlanning - startPlanning)

	####### Accessors #######

	def getSensingRadius(self):
		return self.sensingRadius

	def getRobotGraph(self):
		return self.robot_graph

	def getNumRobots(self):
		return self.robot_graph.getNumNodes()

	def getPositionListTuples(self):
		return self.robot_graph.getNodeLocationList()

	def getPositionList(self):
		posList = []
		for loc in self.robot_graph.getNodeLocationList():
			posList += list(loc)
		return posList

	def getNthEigval(self, n):
		eigvals = math_utils.getListOfAllEigvals(self.relDistanceMatrix)
		eigvals.sort()
		return eigvals[n-1]

	####### Computation #######

	def getGradientOfNthEigenval(self, n):
		# get eigen pair
		nthEigenpair = math_utils.getEigpairLeastFirst(self.relDistanceMatrix, n-1)
		if not (nthEigenpair[0] > 0):
			return False

		gradient = math_utils.getGradientOfMatrixForEigenpair(self.relDistanceMatrix, nthEigenpair, self.robot_graph)
		return gradient

	def moveIsGood(self, move, moveRelative):
		tempSwarm = Swarm(self.sensingRadius)
		if moveRelative:
			curLocs = self.getPositionList()
			newLocs = [sum(i) for i in zip(curLocs, move)]
			tempSwarm.initializeSwarmFromLocationList(newLocs)
		else:
			tempSwarm.initializeSwarmFromLocationList(move)

		if tempSwarm.getNthEigval(4) > 0.75:
			return True

		return False

	def findGoodMove(self):
		grad = self.getGradientOfNthEigenval(4)
		move = []
		for i in range(int(len(grad)/2)):
			if (abs(grad[2*i]) > abs(grad[2*i+1])):
				if (grad[2*i] < 0):
					move += [-1, 0]
				else:
					move += [1, 0]
			else:
				if (grad[2*i+1] < 0):
					move += [0, -1]
				else:
					move += [0, 1]
		return move

	####### Checks #######

	def isRigidFormation(self):
		eigval = self.getNthEigval(4)
		return (not (eigval == 0))

	####### Control #######

	def moveSwarm(self, vector, moveRelative=True):
		self.robot_graph.moveTowardsVec(vector, relativeMovement=moveRelative)

	####### Display Utils #######

	def printAllEigvals(self):
		eigvals = math_utils.getListOfAllEigvals(self.relDistanceMatrix)
		eigvals.sort()
		print(eigvals)

	def printNthEigval(self, n):
		eigvals = math_utils.getListOfAllEigvals(self.relDistanceMatrix)
		eigvals.sort()
		print(eigvals[n-1])

	def showSwarm(self):
		self.robot_graph.displayGraphWithEdges()
