import scipy.spatial
import numpy as np

import math_utils
import math

class Environment():
	
	def __init__(self, bounds, useGrid, numSquaresWide, numSquaresTall):
		self.obstacles = []
		self.bounds = bounds # xlb, xub, ylb, yub
		self.useGrid = useGrid
		self.obstacleKDTree = KDTree(self.getObstacleCentersList())

		if useGrid:
			self.grid = None
			xlb, xub, ylb, yub = bounds
			self.numSquaresWide = numSquaresWide
			self.squareWidth = (xub-xlb)/numSquaresWide
			self.numSquaresTall = numSquaresTall
			self.squareHeight = (yub-ylb)/numSquaresTall

	##### Modify and Initialize ############

	def addObstacle(self, obs):
		self.obstacles.append(obs)
		return None

	def initializeCurvesObstacles(self, numObstacles=50):
		# center = (0.5, 0.5)
		self.curveMaze = True
		xlb, xub, ylb, yub = self.bounds

		radius = .75
		# make left and right walls
		for y in np.linspace(ylb, yub, 100):
			cenLeft = (xlb, y)
			cenRight = (xub, y)
			obs = Obstacle(cenLeft, radius)
			self.addObstacle(obs)
			obs = Obstacle(cenRight, radius)
			self.addObstacle(obs)

		# make top and bottom walls
		for x in np.linspace(xlb, xub, 100):
			cenBot = (x, ylb)
			cenTop = (x, yub)
			obs = Obstacle(cenBot, radius)
			self.addObstacle(obs)
			obs = Obstacle(cenTop, radius)
			self.addObstacle(obs)

		# left divider
		span = yub-ylb
		dividerLen = 1/2
		xCen = xlb+(xub-xlb)/3
		for y in np.linspace(ylb, ylb + dividerLen*span, 100):
			cen = (xCen, y)
			obs = Obstacle(cen, radius)
			self.addObstacle(obs)

		# right divider
		xCen = xlb+2*(xub-xlb)/3
		for y in np.linspace(ylb+dividerLen*span, yub, 100):
			cen = (xCen, y)
			obs = Obstacle(cen, radius)
			self.addObstacle(obs)


		if self.useGrid:
			self.initializeGrid()

	def initializeRandomObstacles(self, numObstacles=50):
		# center = (0.5, 0.5)
		radius = 0.1
		for i in range(numObstacles):
			cnt = 0
			low = min(self.bounds)  
			upp = max(self.bounds)  
			radius = math_utils.genRandomTuple(lb=0, ub=.35, size=1)
			center = math_utils.genRandomTuple(lb=low, ub=upp, size=2)
			while not self.isInsideBounds(center):
				center = math_utils.genRandomTuple(lb=low, ub=upp, size=2)
			obs = Obstacle(center, radius[0])
			self.addObstacle(obs)

		if self.useGrid:
			self.initializeGrid()

	def initializeGrid(self):
		xlb, xub, ylb, yub = self.bounds


		grid = [[] for i in range(self.numSquaresTall)]

		# start top left work across and then down
		curYCenter = ylb + self.squareHeight/2

		row = 0
		while curYCenter < yub:
			curXCenter =  xlb + self.squareWidth/2
			while curXCenter < xub:
				g = GridSquare(self.squareWidth, self.squareHeight, (curXCenter, curYCenter), self.obstacles)
				grid[row].append(g)
				curXCenter += self.squareWidth
			curYCenter += self.squareHeight
			row += 1
			
		self.grid = grid

	###### Check Status ############

	def isFreeSpace(self, coords):
		if (not self.isInsideBounds(coords)):
			return False
		idxs, dist = obstacleKDTree.search(np.array(coords).reshape(2, 1))
		if dist[0] <= self.obstacles[idxs].getRadius():
			return False  # collision
		return True

	def isValidPath(self, startLoc, endLoc):
		if (not self.isInsideBounds(coords)):
			return False
		sx, sy = startLoc
		gx, gy = startLoc
		move = [gx-sx, gy-sy]
		start = [sx, sy]
		for i in range(15):
			loc = [sx+(i+1)/15*move[0], sy+(i+1)/15*move[1]]
			idxs, dist = obstacleKDTree.search(np.array(loc)).reshape(2, 1)
			if dist[0] <= self.obstacles[idxs].getRadius():
				return False  # collision
				
		# for obs in self.obstacles:
		# 	if obs.isInsideObstacle(coords):
		# 		return False
		return True


	def isInsideBounds(self, coords):
		x, y = coords
		xlb, xub, ylb, yub = self.bounds
		return ((xlb < x < xub) and (ylb < y < yub))

	###### Accessors ############

	def getGrid(self):
		assert(self.useGrid)
		assert(self.grid is not None)
		return self.grid

	def getGridBounds(self):
		assert(self.useGrid)
		assert(self.grid is not None)
		return (self.numSquaresWide, self.numSquaresTall)

	def getGridSquareSize(self):
		assert(self.useGrid)
		return (self.squareWidth, self.squareHeight)

	def getObstacleList(self,):
		return self.obstacles
		return None

	def getObstacleCentersList(self,):
		centers = []
		for obs in  self.obstacles:
			cen = list(obs.getCenter())
			centers.append(cen)
		return centers


	def getBounds(self,):
		return self.bounds
		return None

	def getEndOfMazeCenter(self):
		assert(self.curveMaze)
		xlb, xub, ylb, yub = self.bounds
		xCen = xlb+3*(xub-xlb)/4
		yCen = ylb+3*(yub-ylb)/4
		return(xCen, yCen)

	###### Converter ############

	def locationListToGridIndexList(self, locationList):
		assert(self.useGrid)
		gridList = []
		for loc in locationList:
			gridList.append(self.locationToGridIndex(loc))
		return gridList
		
	def locationToGridIndex(self, location):
		if self.isInsideBounds(location):
			xlb, xub, ylb, yub = self.bounds
			x, y = location
			xIndex = math.floor((x - xlb)/self.squareWidth)
			yIndex = math.floor((y - ylb)/self.squareHeight)
			return (xIndex, yIndex)
		else:
			return None

	def gridIndexListToLocationList(self, gridIndexList):
		assert(self.useGrid)
		locList = []
		for gridIndex in gridIndexList:
			locList.append(self.gridIndexToLocation(gridIndex))
		return locList
		
	def gridIndexToLocation(self, gridIndex):
		xInd, yInd = gridIndex
		if (0 <= xInd < self.numSquaresWide) and (0 <= yInd < self.numSquaresTall):
			return self.grid[yInd][xInd].getGridSquareCenter()
		else:
			raise AssertionError


class Obstacle():
	def __init__(self, center, radius):
		self.center = center
		self.radius = radius

	def getCenter(self,):
		return self.center
		return None

	def getRadius(self,):
		return self.radius
		return None

	def isInside(self, coords):
		xpos, ypos = coords
		xcenter, ycenter = self.center
		delta = np.array([xpos-xcenter, ypos-ycenter])
		return (np.linalg.norm(delta, 2) < self.radius)

class GridSquare():

	def __init__(self, width, height, centerCoords, obstacleList):
		cx, cy = centerCoords
		self.x_center_ = cx
		self.y_center_ = cy
		self.width = width
		self.height = height
		self.hasObstacle = self.isGridSquareOccupiedByObstaclesInList(obstacleList)
		self.isOccupied = False


	def setOccupied(self, occStatus):
		assert(not self.hasObstacle) # shouldn't be changing status if has obstacle
		self.isOccupied = occStatus

	def getGridSquareSize(self):
		return (self.width, self.height)
		return None
		
	def getGridSquareCenter(self):
		return (self.x_center_, self.y_center_)
		return None

	def isOccupied(self):
		return self.isOccupied
		return None


	def isSquareFree(self):
		return (not(self.hasObstacle or self.isOccupied))
		return None

	def isGridSquareOccupiedByObstaclesInList(self, obstacles):
		for obs in obstacles:
			if self.isObstacleInGridSquare(obs):
				return True
		return False

	def isObstacleInGridSquare(self, obs):
		cx = self.x_center_
		cy = self.y_center_
		xoff = self.width/2
		yoff = self.height/2
		corners = [(cx-xoff, cy-yoff), (cx-xoff, cy+yoff), (cx+xoff, cy-yoff), (cx+xoff, cy+yoff)]

		for point in corners:
			if obs.isInside(point):
				return True

		return False

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

