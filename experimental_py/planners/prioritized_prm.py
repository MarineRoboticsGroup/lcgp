"""

Probablistic Road Map (PRM) Planner

author: Alan Papalia (@alanpapalia)

"""

import random
import math
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt


class PriorityPrm():

	def __init__(self, robots, env, goals, minEigval=0.75):
		self.robots = robots
		self.env = env.getBounds()
		self.obstacles = env.getObstacleList()
		self.bounds = env.getGridBounds()
		self.goalLocs = goals
		self.startLocs = self.robots.getPositionListTuples()
		self.minEigval = minEigval
		self.obstacleKDTree = KDTree(self.env.getObstacleCentersList())

		self.N_SAMPLE = 1000
		self.N_KNN = 15
		self.MAX_EDGE_LEN = self.robots.getSensingRadius()

	class Node:
		"""
		Node class for dijkstra search
		"""

		def __init__(self, loc, cost, pind):
			self.loc = loc
			self.x = loc[0]
			self.y = loc[1]
			self.cost = cost
			self.pind = pind

		def __str__(self):
			return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)

		def getLocation(self):
			return self.loc


	class KDTree:
		"""
		Nearest neighbor search class with KDTree
		"""

		def __init__(self, data):
			# store kd-tree
			self.tree = scipy.spatial.cKDTree(data)

		def search(self, inp, k=1):
			"""
			Search NN

			inp: input data, single frame or multi frame

			"""

			if len(inp.shape) >= 2:  # multi input
				index = []
				dist = []

				for i in inp.T:
					idist, iindex = self.tree.query(i, k=k)
					index.append(iindex)
					dist.append(idist)

				return index, dist

			dist, index = self.tree.query(inp, k=k)
			return index, dist

		def search_in_distance(self, inp, r):
			"""
			find points with in a distance r
			"""

			index = self.tree.query_ball_point(inp, r)
			return index

	# TODO ALAN
	def planning(self,):
		sampleLocs = self.generateSampleLocations()
		road_map = self.generateRoadmap(sampleLocs, obstacleKDTree)

		trajs = [[] for x in range(robots.getNumRobots())]
		foundGoals = [False for x in range(robots.getNumRobots())]
		curIdx = 0
		while False in foundGoals:
			print("planning for robot", curIdx)
			traj, planSuccess = self.astar_planning(road_map, sampleLocs, curIdx)
			if planSuccess:
				trajs[curIdx] = traj
				foundGoals[curIdx] = True
				curIdx += 1
			else:
				raise NotImplementedError
				# Identify constraint
				# Propagate constraint
				# Figure out the constraint driven search
				foundGoals[curIdx] = True
				curIdx -= 1

		return trajs

	def isValidPath(self, curLoc, connectingLoc):
		dx = curLoc[0] - connectingLoc[0]
		dy = curLoc[1] - connectingLoc[1]
		dist = math.hypot(dx, dy)

		# node too far away
		if dist >= MAX_EDGE_LEN:
			return False
		return self.env.isValidPath(curLoc, connectingLoc)

	def generateRoadmap(self, sampleLocs):
		"""
		Road map generation

		sampleLocs: [[m, m]...[m,m]] x-y positions of sampled points

		@return: list of list of edge ids ([[edges, from, 0], ...,[edges, from, N])
		"""

		road_map = []
		nsample = len(sampleLocs)
		sampleKDTree = KDTree(sampleLocs)

		for curLoc in sampleLocs:

			index, dists = sampleKDTree.search(np.array(curLoc).reshape(2, 1), k=self.N_KNN)
			inds = index[0]
			edge_id = []

			for ii in range(1, len(inds)):
				connectingLoc = sampleLocs[inds[ii]]

				if self.isValidPath(curLoc, connectingLoc):
					edge_id.append(inds[ii])

				if len(edge_id) >= N_KNN:
					break

			road_map.append(edge_id)

		 plot_road_map(road_map, sampleLocs)

		return road_map

	# TODO ALAN
	def astar_planning(self, road_map, sampleLocs, curRobotId):
		"""
		road_map: ??? [m]
		sampleLocs: ??? [(m, m), .., (m, m)]

		@return: list of node ids ([id0, id1, ..., idN]), empty list when no path was found
		"""

		nstart = Node(self.startLocs, 0.0, -1)
		ngoal = Node(self.goalLocs, 0.0, -1)

		openset, closedset = dict(), dict()
		openset[len(road_map) - 2] = nstart

		path_found = True

		while True:
			if not openset:
				print("Cannot find path")
				path_found = False
				break

			# find minimum cost in openset
			curr_id = min(open_set, key=lambda o: open_set[o].cost + calcHeuristic(open_set[o], ngoal))
			# curr_id = min(openset, key=lambda o: openset[o].cost)
			currNode = openset[curr_id]
			curLoc = currNode.getLocation()

			# show graph
			# if show_animation and len(closedset.keys()) % 2 == 0:
			# 	# for stopping simulation with the esc key.
			# 	plt.gcf().canvas.mpl_connect('key_release_event',
			# 			lambda event: [exit(0) if event.key == 'escape' else None])
			# 	plt.plot(currNode.x, currNode.y, "xg")
			# 	plt.pause(0.001)

			if curr_id == (len(road_map) - 1):
				print("goal is found!")
				ngoal.pind = currNode.pind
				ngoal.cost = currNode.cost
				break

			# Remove the item from the open set
			del openset[curr_id]
			# Add it to the closed set
			closedset[curr_id] = currNode

			# expand search grid based on motion model
			for i in range(len(road_map[curr_id])):
				new_id = road_map[curr_id][i]
				if new_id in closedset:
					continue
								
				newLoc = sampleLocs[new_id]
				dist = calcDistanceBetweenLocations(curLoc, newLoc)
				newNode = Node(newLoc, currNode.cost + dist, curr_id)

				# Otherwise if it is already in the open set
				if new_id in openset:
					if openset[new_id].cost > newNode.cost:
						openset[new_id].cost = newNode.cost
						openset[new_id].pind = curr_id
				else:
					openset[new_id] = newNode

		if path_found is False:
			return ([], False)

		# generate final course
		pathIdxs = [len(road_map)-1] # Add goal node
		pind = ngoal.pind
		while pind != -1:
			pathIdxs.append(pind)
			n = closedset[pind]
			pind = n.pind

		return (pathIdxs, True)


	def plot_road_map(road_map, sampleLocs):  # pragma: no cover

		for i, _ in enumerate(road_map):
			for ii in range(len(road_map[i])):
				ind = road_map[i][ii]

				plt.plot([sampleLocs[i][0], sampleLocs[ind][0]],
						 [sampleLocs[i][1], sampleLocs[ind][1]], "-k")

	def generateSampleLocations(self, ):

		xlb, xub, ylb, yub = self.bounds

		sampleLocs = []

		while len(sampleLocs) <= N_SAMPLE:
			tx, ty = math_utils.genRandomLocation(xlb, xub, ylb, yub) 

			index, dist = obstacleKDTree.search(np.array([tx, ty]).reshape(2, 1))

			# If not within obstacle
			if dist[0] >= self.obstacles[index].getRadius():
				sampleLocs.append([tx, ty])

		sampleLocs.append([sx, sy])
		sampleLocs.append([gx, gy])

		return sampleLocs


	def calcHeuristic(curNode, goalNode):
		curLoc = curNode.getLocation()
		goalLoc = goalNode.getLocation()
		return calcDistance(curLoc, goalLoc)

	def calcDistanceBetweenLocations(loc1, loc2):
		nx, ny = loc1
		gx, gy = loc2
		dx = gx-nx
		dy = gy-ny
		return math.hypot(dx, dy)
