import numpy as np
import math_utils
import math

class Environment():
	
	def __init__(self, bounds, useGrid, numSquaresWide, numSquaresTall):
		self.obstacles = []
		self.bounds = bounds # xlb, xub, ylb, yub
		self.useGrid = useGrid
		
		if useGrid:
			self.grid = None
			xlb, xub, ylb, yub = bounds
			self.numSquaresWide = numSquaresWide
			self.squareWidth = (xub-xlb)/numSquaresWide
			self.numSquaresTall = numSquaresTall
			self.squareHeight = (yub-ylb)/numSquaresTall

	###### Modify and Initialize ############

	def addObstacle(self, obs):
		self.obstacles.append(obs)

	def initializeRandom(self, numObstacles=50):
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
		for obs in self.obstacles:
			if obs.isInsideObstacle(coords):
				return False
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

	def getBounds(self,):
		return self.bounds

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
			return [-1, -1]

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

	def getRadius(self,):
		return self.radius

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

	###### Accessors ############## 

	def getGridSquareSize(self):
		return (self.width, self.height)
		
	def getGridSquareCenter(self):
		return (self.x_center_, self.y_center_)

	def isOccupied(self):
		return self.isOccupied

	def isSquareFree(self):
		return (not(self.hasObstacle or self.isOccupied))

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
