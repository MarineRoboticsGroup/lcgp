import numpy as np

import graph
import math_utils

class Swarm:

	def __init__(self, sensingRadius):
		self.sensingRadius = sensingRadius
		self.robot_graph = graph.Graph()

	def initializeSwarm(self):
		# intialize formation and edges
		self.robot_graph.removeAllNodes()
		self.robot_graph.removeAllEdges()
		self.robot_graph.initializeSquare()
		# self.robot_graph.initializeRigidLine()
		self.updateSwarm()

		# Get values and representations
		self.relDistanceMatrix = self.robot_graph.getStiffnessMatrix()

	def updateSwarm(self):
		self.robot_graph.updateEdgesByRadius(self.sensingRadius)
		self.relDistanceMatrix = self.robot_graph.getStiffnessMatrix()

	def getGradientOfNthEigenval(self, n):
		# get eigen pair
		nthEigenpair = math_utils.getEigpairLeastFirst(self.relDistanceMatrix, n-1)
		if not (nthEigenpair[0] > 0):
			return False

		gradient = math_utils.getGradientOfMatrixForEigenpair(self.relDistanceMatrix, nthEigenpair, self.robot_graph)
		return gradient


	def moveSwarm(self, vector):
		self.robot_graph.moveTowardsVec(vector, relativeMovement=True)


	####### Display Utils #######

	def printAllEigvals(self):
		eigvals = math_utils.getListOfAllEigvals(self.relDistanceMatrix)
		eigvals.sort()
		print(eigvals)

	def printNthEigval(self, n):
		eigvals = math_utils.getListOfAllEigvals(self.relDistanceMatrix)
		eigvals.sort()
		print(eigvals[n-1])

	def getNthEigval(self, n):
		eigvals = math_utils.getListOfAllEigvals(self.relDistanceMatrix)
		eigvals.sort()
		return eigvals[n-1]


	def showSwarm(self):
		self.robot_graph.displayGraphWithEdges()
