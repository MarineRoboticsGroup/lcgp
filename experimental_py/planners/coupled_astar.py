import copy
import math
from itertools import permutations
import itertools

import graph

class CoupledAstar():
	def __init__(self, robots, env, goals, minEigval=0.75):
		self.robots = robots
		self.env = env
		self.grid = env.getGrid()
		self.gridBounds = env.getGridBounds()
		self.goalIndexs = goals
		self.startIndexs = self.robots.getPositionListTuples()
		self.minEigval = minEigval
		self.motionModel = self.getMotionModel(self.robots.getNumRobots(), self.env.getGridSquareSize())

	class CoupledAstarNode:
		def __init__(self, gridIndexs, cost, pind):
			self.gridIndexs = gridIndexs
			self.cost = cost
			self.pind = pind

		def getNodeHash(self):
			for i, indexs in enumerate(self.gridIndexs):
				if i is 0:
					h = indexs
				else:
					h += indexs
			return h

		def getGridIndexList(self):
			return self.gridIndexs

	def planning(self,):
		nstart = self.CoupledAstarNode(self.startIndexs, 0.0, -1)
		ngoal = self.CoupledAstarNode(self.goalIndexs, 0.0, -1)

		goal_id = ngoal.getNodeHash()

		open_set, closed_set = dict(), dict()
		open_set[nstart.getNodeHash()] = nstart

		while 1:
			if len(open_set) == 0:
				print("Open set is empty..")
				break

			c_id = min(open_set, key=lambda o: open_set[o].cost + self.calcHeuristic(open_set[o], ngoal))
			curNode = open_set[c_id]

			if self.isGoal(curNode):
				print("Find goal")
				ngoal.pind = curNode.pind
				ngoal.cost = curNode.cost
				break

			# Remove the item from the open set
			del open_set[c_id]

			# Add it to the closed set
			closed_set[c_id] = curNode

			# expand_grid search grid based on motion model
			curLocs = curNode.getGridIndexList()
			for motions in self.motionModel:
				# make new node
				newLocs = []
				for i, move in enumerate(motions[0]):
					newLocs.append(tuple((curLocs[i][0]+move[0], curLocs[i][1]+move[1])))
				node = self.CoupledAstarNode(newLocs, curNode.cost + motions[1], c_id)
				n_id = node.getNodeHash()

				# If the node is not safe, do nothing
				if not self.verify_node(node):
					continue

				if n_id in closed_set:
					continue

				if n_id not in open_set:
					open_set[n_id] = node  # discovered a new node
				else:
					if open_set[n_id].cost > node.cost:
						# This path is the best until now. record it
						open_set[n_id] = node

		return self.calcFinalPath(ngoal, closed_set)

	def calcFinalPath(self, ngoal, closed_set):
		locPath = [[] for i in range(self.robots.getNumRobots())]

		locs = ngoal.getGridIndexList()
		for i, loc in enumerate(locs):
			locPath[i].append(loc)

		pind = ngoal.pind
		while pind != -1:
			n = closed_set[pind]
			locs = n.getGridIndexList()
			for i, loc in enumerate(locs):
				locPath[i].append(loc)
			pind = n.pind
		for path in locPath:
			path.reverse()
		return locPath

	def calcHeuristic(self, curNode, goalNode):
		curIndexList = curNode.getGridIndexList()
		goalIndexList = goalNode.getGridIndexList()
		return self.sumOfDistancesBetweenGridIndexLists(curIndexList, goalIndexList)

	@staticmethod
	def getMotionModel(nRobots, gridSizes):
		# dx, dy, cost
		temp_motions = []
		temp_motions_i = list(set(permutations([1, 1, 0, -1, -1], 2)))
		for i in range(nRobots):
			if i is 0:
				temp_motions = temp_motions_i
			else:
				temp_motions = (itertools.product(temp_motions, temp_motions_i))

		for i in range(nRobots-2):
			temp_motions = [(*x[0], *x[1:]) for x in temp_motions]

		gridW, gridH = gridSizes
		for i, moveList in enumerate(temp_motions):

			dists = []
			for move in moveList:
				deltax = move[0]*gridW
				deltay = move[1]*gridH
				dist = math.hypot(deltax, deltay)
				dists.append(dist)
			distMove = sum(dists)
			temp_motions[i] = (list(moveList), distMove)

		return temp_motions

	def verify_node(self, node):
		locList = node.getGridIndexList()
		# check if inside bounds
		for i, loc in enumerate(locList):
			xind, yind = loc

			# check inside bounds
			if (xind < 0) or (yind < 0) or (xind >= self.gridBounds[0]) or (yind >= self.gridBounds[1]):
				return False

			# check if free space
			if not self.grid[xind][yind].isSquareFree():
				return False

			# check for collisions
			for otherloc in locList[i+1:]:
				if loc == otherloc:
					return False

		# check geometry
		# g = graph.Graph()
		# g.initializeFromLocationList(locList, self.robots.getSensingRadius())
		# eigval = g.getNthEigval(4)
		# if eigval < self.minEigval:
		# 	return False

		return True

	def sumOfDistancesBetweenGridIndexLists(self, locationList, goalList):
		assert(len(locationList) == len(goalList))
		dists = []
		for i, _ in enumerate(locationList):
			dists.append(self.distanceBetweenLocs(locationList[i], goalList[i]))
		return sum(dists)

	def distanceBetweenLocs(self, loc1, loc2):
		ix1, iy1 = loc1
		ix2, iy2 = loc2
		x1, y1 = self.grid[ix1][iy1].getGridSquareCenter()
		x2, y2 = self.grid[ix2][iy2].getGridSquareCenter()
		deltax = x1-x2
		deltay = y1-y2
		return math.hypot(deltax, deltay)

	def isGoal(self, node):
		return (node.getGridIndexList() == self.goalIndexs)