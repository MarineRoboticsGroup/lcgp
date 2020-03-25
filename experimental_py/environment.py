class Environment():
	def __init__(self, ):
		self.obstacles = []
	
	def addObstacle(Obstacle):		
		self.obstacles.append()

	def isFreeSpace(coords):
		for obs in self.obstacles:
			if obs.isInsideObstacle(coords):
				return False
		return True

class RectObstacle():
	def __init__(self, bounds):
		self.boundaries = bounds

	def isInside(self, coords):
		xpos, ypos = coords
		left, right, upper, lower = self.boundaries
		return ((left <= xpos <= right) and (lower <= ypos <= upper))


