"""

Probablistic Road Map (PRM) Planner

author: Alan Papalia (@alanpapalia)

"""

import random
import math
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import os.path
from os import path
import string
import time

import math_utils
import plot
import swarm
import kdtree

class PriorityPrm():

	class Node:
		"""
		Node class for dijkstra search
		"""

		def __init__(self, loc, cost, pind, timestep, index, useTime):
			self.loc = loc
			self.x = loc[0]
			self.y = loc[1]
			self.cost = cost
			self.pind = pind
			self.timestep = timestep
			self.index = index
			if useTime:
				self.pkey = (pind, timestep-1)
			else:
				self.pkey = pind



		def __str__(self):
			return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)

		def getLocation(self):
			return self.loc

		def setTimestep(self, timestep):
			self.timestep = timestep

	def __init__(self, robots, env, goals, maxsteps=99999):
		self.robots = robots
		self.sensingRadius = self.robots.getSensingRadius()
		self.startLocs = self.robots.getPositionListTuples()
		self.numRobots = robots.getNumRobots()
		self.startConfig = self.robots.startConfig

		self.env = env
		self.obstacles = env.getObstacleList()
		self.bounds = env.getBounds()
		self.goalLocs = goals
		
		self.minEigval = robots.minEigval
		self.maxsteps = maxsteps

		self.N_SAMPLE = 3000
		self.N_KNN = 30
		self.MAX_EDGE_LEN = 2
		self.roadmapFilename = 'roadmap_%s_%dsamples_%dnn_%dlen_%drob.txt'%(self.startConfig, self.N_SAMPLE, self.N_KNN, self.MAX_EDGE_LEN, self.numRobots)
		self.nodeKDTree = None



	# TODO: constraint handling and propogation
	def planning(self, addTimeDimension=False):

		print("Sampling Locations")
		sampleLocs = self.generateSampleLocations()
		self.nodeKDTree = kdtree.KDTree(sampleLocs)
		print("%d Locations Sampled\n"%len(sampleLocs))
		
		roadmap = self.readRoadmap()
		if roadmap and (len(roadmap) > 0):
			print("Read from existing roadmap file: %s\n"%self.roadmapFilename)
		else:
			print("%s not found.\nGenerating Roadmap"%self.roadmapFilename)
			roadmap = self.generateRoadmap(sampleLocs)
			self.writeRoadmap(roadmap)
			print("New roadmap written to file\n")

		# self.plotRoadmap(roadmap, sampleLocs)
		# plot.showPlot()

		print("Beginning Planning")
		print()

		feasTraj = self.initFeasTraj(roadmap)

		trajs = [[] for x in range(self.robots.getNumRobots())]
		coordTrajs = [[] for x in range(self.robots.getNumRobots())]
		foundGoals = [False for x in range(self.robots.getNumRobots())]
		curIdx = 0
		while False in foundGoals:
			print("Planning for robot", curIdx)
			timeStart = time.time()
			traj, conflict = self.astarPlanning(roadmap, sampleLocs, curIdx, trajs, useTime=addTimeDimension)
			timeEnd = time.time()
			print("Planning completed in: %f (s)"%(timeEnd-timeStart))
			if conflict is None:
				print("Planning succesful for robot %d \n"%curIdx)
				trajs[curIdx] = traj
				coordTrajs[curIdx] = self.convertTrajectoryToCoords(traj, sampleLocs)
				foundGoals[curIdx] = True
				curIdx += 1

				plot.showTrajectories(coordTrajs, self.robots, self.env, self.goalLocs)
			else:
				print("Planning Failed for robot %d. \nPropagating Constraint\n"%curIdx)
				raise NotImplementedError
				# Identify constraint
				# Propagate constraint
				# Figure out the constraint driven search
				foundGoals[curIdx] = True
				curIdx -= 1


		trajs = self.convertTrajectoriesToCoords(trajs, sampleLocs)
		print("Full set of trajectories found!")
		return trajs

	# TODO: add conflict checking
	# TODO: way of returning conflict
	def astarPlanning(self, roadmap, sampleLocs, curRobotId, currTrajs, useTime=False):
		"""
		roadmap: ??? [m]
		sampleLocs: ??? [(m, m), .., (m, m)]

		@return: list of node ids ([id0, id1, ..., idN]), empty list when no path was found
		"""
		startId = len(roadmap) - 2*self.numRobots+curRobotId
		goalId = len(roadmap) - self.numRobots+curRobotId
		print("StartID", startId, sampleLocs[startId])
		print("GoalID", goalId, sampleLocs[goalId])
		print()
		# loc cost pind timestep index
		startNode = self.Node(self.startLocs[curRobotId], cost=0.0, pind=-1, timestep=0, index=startId, useTime=useTime)
		goalNode = self.Node(self.goalLocs[curRobotId], cost=0.0, pind=-1, timestep=-1, index=goalId, useTime=useTime)

		openSet, closedSet = dict(), dict()
		openSet[self.getNodeKey(startNode, useTime)] = startNode

		conflict = None

		while True:


			# if out of options, return conflict information
			if not openSet:
				self.plotFailedSearch(closedSet, sampleLocs, roadmap)
				conflict = True
				return ([], conflict)

			# find minimum cost in openSet
			currKey = min(openSet, key=lambda o: openSet[o].cost + self.calcHeuristic(openSet[o], goalNode, useTime))

			currNode = openSet[currKey]

			# Remove the item from the open set
			del openSet[currKey]
			closedSet[currKey] = currNode

			# if goal location
			if self.foundGoal(currNode, goalNode):
				goalNode.pind = currNode.pind
				goalNode.cost = currNode.cost
				goalNode.timestep = currNode.timestep
				goalNode.pkey = currNode.pkey
				break


			# If node is valid continue to expand path
			if self.nodeIsValid(currNode, currTrajs, sampleLocs, curRobotId):
				# Add it to the closed set
				closedSet[currKey] = currNode
				curLoc = currNode.getLocation()

				# expand search grid based on motion model
				for i in range(len(roadmap[currNode.index])):
					new_id = roadmap[currNode.index][i]
					if new_id in closedSet:
						continue
									
					newLoc = sampleLocs[new_id]
					dist = self.calcDistanceBetweenLocations(curLoc, newLoc)
					newNode = self.Node(loc=newLoc, cost=currNode.cost + dist, pind=currNode.index, timestep=currNode.timestep+1, index=new_id, useTime=useTime)
					newKey = self.getNodeKey(newNode, useTime)

					# Otherwise if it is already in the open set
					if newKey in openSet:
						if openSet[newKey].cost > newNode.cost:
							# openSet[newKey] = newNode
							openSet[newKey].cost = newNode.cost
							openSet[newKey].pind = newNode.pind
							openSet[newKey].timestep = newNode.timestep
							openSet[newKey].pkey = newNode.pkey
					else:
						openSet[newKey] = newNode

		# generate final course
		pathIdxs = [len(roadmap)-self.numRobots+curRobotId] # Add goal node
		pkey = goalNode.pkey
		pind = goalNode.pind
		while pind != -1:
			pathIdxs.append(pind)
			node = closedSet[pkey]
			pkey = node.pkey
			pind = node.pind

		pathIdxs.reverse()
		return (pathIdxs, None)

	def isValidPath(self, curLoc, connectingLoc):
		dx = curLoc[0] - connectingLoc[0]
		dy = curLoc[1] - connectingLoc[1]
		dist = math.hypot(dx, dy)

		# node too far away
		if dist >= self.MAX_EDGE_LEN:
			return False
		return self.env.isValidPath(curLoc, connectingLoc)

	def generateRoadmap(self, sampleLocs):
		"""
		Road map generation

		sampleLocs: [[m, m]...[m,m]] x-y positions of sampled points

		@return: list of list of edge ids ([[edges, from, 0], ...,[edges, from, N])
		"""
		roadmap = []
		nsample = len(sampleLocs)
		sampleKDTree = self.nodeKDTree

		for curLoc in sampleLocs:

			index, dists = sampleKDTree.search(np.array(curLoc).reshape(2, 1), k=self.N_KNN)
			inds = index[0]
			edge_id = []

			for ii in range(1, len(inds)):
				connectingLoc = sampleLocs[inds[ii]]

				if self.isValidPath(curLoc, connectingLoc):
					edge_id.append(inds[ii])

				if len(edge_id) >= self.N_KNN:
					break

			roadmap.append(edge_id)
		return roadmap

	def generateSampleLocations(self, ):
		xlb, xub, ylb, yub = self.bounds
		sampleLocs = []
		while len(sampleLocs) < self.N_SAMPLE:
			newLoc = math_utils.genRandomLocation(xlb, xub, ylb, yub) 

			# If not within obstacle
			if self.env.isFreeSpace(newLoc):
				sampleLocs.append(list(newLoc))

		for loc in self.startLocs:
			sampleLocs.append(list(loc))
		for loc in self.goalLocs:
			sampleLocs.append(list(loc))

		return sampleLocs

	def nodeIsValid(self, currNode, currTrajs, sampleLocs, curRobotId):
		timestep = currNode.timestep
		if timestep > self.maxsteps:
			# print("Timestep is past allowable range")
			return False

		for roboId in range(curRobotId):
			lastStep = len(currTrajs[roboId])-1
			locId = currTrajs[roboId][min(timestep, lastStep)]
			if locId == currNode.index:
				# print("%d cannot occupy same space as %d"%(curRobotId,roboId) )
				return False

		if curRobotId == 0:
			return True
		if curRobotId == 1:
			lastStep = len(currTrajs[0])-1
			index1 = currTrajs[0][min(timestep, lastStep)]
			loc1 = sampleLocs[index1]
			loc2 = sampleLocs[currNode.index]
			if (1 < self.calcDistanceBetweenLocations(loc1, loc2) < self.sensingRadius):
				return True
			else:
				# print("%d would be too far from %d"%(curRobotId,roboId) )
				return False
		if curRobotId >= 2:
			locList = self.getAssignedLocations(currTrajs, sampleLocs, timestep, curRobotId)
			locList.append(sampleLocs[currNode.index])
			minEigval = self.getMinEigenvalFromLocList(locList)
			if (minEigval > self.minEigval):
				return True
			else:
				# if minEigval == 0:
				# 	print("The matrix would be flexible")
				# else:
				# 	print("The matrix would not be sufficiently rigid:", minEigval)
				# print()
				return False

	def getMinEigenvalFromLocList(self, locList):
		tempSwarm = swarm.Swarm(self.sensingRadius)
		tempSwarm.initializeSwarmFromLocationListTuples(locList)
		return tempSwarm.getNthEigval(4)

	@staticmethod
	def getNodeKey(node, useTime):
		if useTime:
			return (node.index, node.timestep)
		else:
			return node.index

	@staticmethod
	def foundGoal(currNode, goalNode):
		return (currNode.index == goalNode.index)

	@staticmethod
	def setGoalToCurrent(currNode, goalNode):
		goalNode.pind = currNode.pind
		goalNode.pkey = currNode.pkey
		goalNode.cost = currNode.cost
		goalNode.timestep = currNode.timestep

	@staticmethod
	def getAssignedLocations(currTrajs, sampleLocs, timestep, curRobotId):
		locList = []
		for i in range(curRobotId):
			lastStep = len(currTrajs[i])-1
			locList.append(sampleLocs[currTrajs[i][min(timestep, lastStep)]])
		return locList

	@staticmethod
	def convertTrajectoriesToCoords(trajs, sampleLocs):
		newTrajs = []
		for traj in trajs:
			newTraj = []
			for index in traj:
				newTraj.append(sampleLocs[index])
			newTrajs.append(newTraj)
		return newTrajs

	@staticmethod
	def convertTrajectoryToCoords(traj, sampleLocs):
		coords = []
		for index in traj:
			coords.append(sampleLocs[index])
		return coords

	@staticmethod
	def plotRoadmap(roadmap, sampleLocs):  # pragma: no cover
		print("Displaying Roadmap... May take time :)")
		edges = set()
		for i, _ in enumerate(roadmap):
			for ii in range(len(roadmap[i])):
				ind = roadmap[i][ii]
				
				edge = (ind, i) if ind < i else (i, ind) 
				if edge in edges:
					continue
				else:
					edges.add(edge)
					plt.plot([sampleLocs[i][0], sampleLocs[ind][0]],
							 [sampleLocs[i][1], sampleLocs[ind][1]], "-k")

	@staticmethod
	def calcHeuristic(curNode, goalNode, useTime):
		curLoc = curNode.getLocation()
		goalLoc = goalNode.getLocation()
		nx, ny = curLoc
		gx, gy = goalLoc
		dx = gx-nx
		dy = gy-ny
		if useTime:
			return math.hypot(dx, dy) + 5*curNode.timestep
		else:
			return math.hypot(dx, dy)

	@staticmethod
	def calcDistanceBetweenLocations(loc1, loc2):
		nx, ny = loc1
		gx, gy = loc2
		dx = gx-nx
		dy = gy-ny
		return math.hypot(dx, dy)

	def initFeasTraj(self, roadmap):
		feasTraj = [[] for x in range(self.numRobots)]

		# for every robot create feasibility path
		for curRobotId in range(self.numRobots):
			startNodeId = len(roadmap) - 2*self.numRobots + curRobotId
			feasTraj[curRobotId].append(set())
			feasTraj[curRobotId][0].add(startNodeId)
			
			openSet = set([startNodeId])
			connSet = set()

			# continue until robot is capable of reaching all locations
			# Note: Assumes connected roadmap
			while len(openSet) < len(roadmap):

				# for each node in current state add connected nodes
				for nodeId in openSet:
					connSet.update(roadmap[nodeId])

				# clear current state, update feasibility
				openSet.clear()
				feasTraj[curRobotId].append(set())
				feasTraj[curRobotId][-1].update(connSet)
				openSet.update(connSet)
				connSet.clear()
		return feasTraj

	def plotFailedSearch(self, closedSet, sampleLocs, roadmap):
		nodes = closedSet.values()
		for node in nodes:
			if node.pind == -1:
				continue
			path = [sampleLocs[node.index], sampleLocs[node.pind]]
			plt.plot(*zip(*path), color='b')
		plot.plotObstacles(self.env)
		plot.plotGoals(self.goalLocs)
		plot.showPlot()

	@staticmethod
	def getConnectingNodesRoadmap(roadmap, index):
		return roadmap[index]

	def readRoadmap(self,):
		if not path.exists(self.roadmapFilename):
			return False
		rmap = []
		with open(self.roadmapFilename, 'r') as filehandle:
			for line in filehandle:
				roads = list(map(int, line.split()))
				rmap.append(roads)

		return rmap

	def writeRoadmap(self, roadmap):
		with open(self.roadmapFilename, 'w') as filehandle:
			for roads in roadmap:
				line = str(roads).translate(str.maketrans('', '', string.punctuation))
				filehandle.write('%s\n' % line)


