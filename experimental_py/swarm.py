import numpy as np

import graph
import math_utils

class Swarm:

	def __init__(self, sensingRadius):
		self.sensingRadius = sensingRadius
		self.robot_graph = graph.Graph()

	####### Swarm Utils #######

	def initializeSwarm(self, formation='square', bound=10, seed=0):
		# intialize formation and edges
		self.robot_graph.removeAllNodes()
		self.robot_graph.removeAllEdges()

		if formation.lower() == 'square':
			self.robot_graph.initializeSquare()
		elif formation.lower() == 'line':
			self.robot_graph.initializeLine()
		elif formation.lower() == 'rigid-line':
			self.robot_graph.initializeRigidLine()
		elif formation.lower() == 'random':
			self.robot_graph.initializeRandomConfig(bound, seed)
		else:
			print("The given formation is not valid")
			raise AssertionError

		self.updateSwarm()
		self.relDistanceMatrix = self.robot_graph.getStiffnessMatrix()

	def updateSwarm(self):
		self.robot_graph.updateEdgesByRadius(self.sensingRadius)
		self.relDistanceMatrix = self.robot_graph.getStiffnessMatrix()

	####### Accessors #######

	def getSensingRadius(self):
		return self.sensingRadius

	def getRobotGraph(self):
		return self.robot_graph

	def getNumRobots(self):
		return self.robot_graph.getNumNodes()

	def getPositionList(self):
		return self.robot_graph.getNodeLocationList()

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
