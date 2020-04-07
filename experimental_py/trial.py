import sys
sys.path.insert(1, '/home/alan/range_only_robotics/experimental_py/planners')
import matplotlib.pyplot as plt
import numpy as np
import copy


import math_utils
import swarm
import environment
import plot

# planners
import decoupled_rrt
import coupled_astar


def checkFeasibility(swarm, env, goals):
	startEigval = swarm.getNthEigval(4)
	startLocs = swarm.getPositionList()
	graph = swarm.getRobotGraph()
	plot.plotNoGrid(graph, env, goals)

	goalLoc = []
	for goal in goals:
		goalLoc += list(goal)

	swarm.moveSwarm(goalLoc, moveRelative=False)
	swarm.updateSwarm()
	graph = swarm.getRobotGraph()
	goalEigval = swarm.getNthEigval(4)

	# plot.plotNoGrid(graph, env, goals)

	swarm.moveSwarm(startLocs, moveRelative=False)
	swarm.updateSwarm()

	if (0 == startEigval):
		print("Starting Config Invalid")
		print()
		return False
	elif (0 == goalEigval):
		print("Goal Config Invalid")
		print()
		return False
	else:
		return True

def readTrajFromFile(filename, useGrid=False):
	# define an empty list
	trajs = []

	# open file and read the content in a list
	with open(filename, 'r') as filehandle:
		for line in filehandle:
			traj = []

			line = line[:-1]
			# remove linebreak which is the last character of the string
			startInd = line.find('(')
			endInd = line.find(')')
			
			while(startInd != -1):
				sect = line[startInd+1:endInd]
				commaInd = sect.find(',')
				if useGrid:
					xcoord = int(sect[:commaInd])
					ycoord = int(sect[commaInd+1:])
				else:
					xcoord = float(sect[:commaInd])
					ycoord = float(sect[commaInd+1:])
				coords = (xcoord, ycoord)
				traj.append(coords)

				line = line[endInd+1:]
				startInd = line.find('(')
				endInd = line.find(')')

			trajs.append(traj)
	return trajs

def convertAbsoluteTrajToRelativeTraj(locLists):
	relMoves = [[(0,0)] for i in locLists]
	
	for robotNum in range(len(locLists)):
		for i in range(len(locLists[robotNum])-1):
			xold, yold = locLists[robotNum][i]
			xnew, ynew = locLists[robotNum][i+1]
			deltax = xnew-xold 
			deltay = ynew-yold 
			relMoves[robotNum].append((deltax, deltay))
	return relMoves

def testTrajectory(robots, env, trajs, goals, trajNum=1, useGrid=False, delayAnimationStart=False, relativeTraj=False):
	"""
	Takes a generic input trajectory of absolute states 
	and moves the swarm through the trajectory

	:param      robots:               The robots
	:type       robots:               Swarm object
	:param      env:                  The environment
	:type       env:                  Environment object
	:param      trajs:                The trajs
	:type       trajs:                list of lists of tuples of doubles
	:param      goals:                The goals
	:type       goals:                List of tuples
	:param      trajNum:              The traj number
	:type       trajNum:              integer
	:param      delayAnimationStart:  Whether to delay the animation beginning
	:type       delayAnimationStart:  boolean
	"""

	firstImage = delayAnimationStart
	if trajs is None:
		print("Cannot find path")
	else:
		trajIndex = [-1 for traj in trajs]
		finalTrajIndex = [len(traj)-1 for traj in trajs]
		move = []
		minEigvals = []

		while not (trajIndex == finalTrajIndex):
			move.clear()
			for robotIndex in range(robots.getNumRobots()):
				
				# Increment trajectory for unfinished paths
				if trajIndex[robotIndex] != finalTrajIndex[robotIndex]:
					trajIndex[robotIndex] += 1

				# Get next step on paths
				if relativeTraj and trajIndex[robotIndex] == finalTrajIndex[robotIndex]:
					newLoc = (0,0)
				else:
					newLoc = trajs[robotIndex][trajIndex[robotIndex]]
				
				# If in grid mode convert indices to locations
				if useGrid:
					newLoc = env.gridIndexToLocation(newLoc)
				
				move += list(newLoc)

			while not (robots.moveIsGood(move, moveRelative=relativeTraj)):
				move = [x for x in robots.findGoodMove()]
				print("Fixing move!!")

			robots.moveSwarm(move, moveRelative=relativeTraj)
			robots.updateSwarm()

			graph = robots.getRobotGraph()

			minEigval = robots.getNthEigval(4)
			if minEigval < 0.5:
				plot.plotNoGrid(graph, env, goals)

			minEigvals.append(minEigval)
			
			if firstImage:
				plt.pause(10)
				firstImage = False

			if useGrid:
				plot.animationWithGrid(graph, env, goals)
			else:
				plot.animationNoGrid(graph, env, goals)


		plt.close()

		plt.plot(minEigvals)
		plt.hlines([0.2, 0.8], 0, len(minEigvals))
		plt.title("Minimum Eigenvalue over Time")
		plt.ylabel("Eigenvalue")
		plt.xlabel("time")
		plt.show()

		# with open('recent_traj.txt', 'w') as filehandle:
		# 	for traj in trajs:
		# 		filehandle.write('%s\n' % traj)
				

def makeSensitivityPlotsRandomMotions(robots, environment):
	"""
	Makes sensitivity plots random motions.

	:param      robots:       The robots
	:type       robots:       { type_description }
	:param      environment:  The environment
	:type       environment:  { type_description }
	"""
	vectorLength = 0.1

	for vectorLength in [0.15, 0.25, 0.5]:
		robots.initializeSwarm()

		predChanges = []
		actChanges = []
		predRatios = []

		predChangesCrit = []
		actChangesCrit = []
		predRatiosCrit = []

		for i in range(500):

			origEigval = robots.getNthEigval(4)
			grad = robots.getGradientOfNthEigenval(4)
			if grad is False:
				# robots.showSwarm()
				break
			else:
				dirVector = math_utils.genRandomVector(len(grad), vectorLength)
				predChange = np.dot(grad, dirVector)
				if origEigval < 1:
					while (predChange < vectorLength*.8):
						dirVector = math_utils.genRandomVector(len(grad), vectorLength)
						predChange = np.dot(grad, dirVector)
				

				robots.moveSwarm(dirVector)
				robots.updateSwarm()
				newEigval = robots.getNthEigval(4)
				actChange = newEigval - origEigval

				predRatio = actChange/predChange

				if abs(predChange) > 1e-4 and abs(actChange) > 1e-4:
					predChanges.append(predChange)
					actChanges.append(actChange)
					predRatios.append(predRatio)

					if origEigval < 1:
						predChangesCrit.append(predChange)
						actChangesCrit.append(actChange)
						predRatiosCrit.append(predRatio)


		if len(predChanges) > 0:
			# plot ratio
			print("Making plots for step size:", vectorLength, "\n\n")
			plt.figure()
			plt.plot(predRatios)	
			plt.ylim(-3, 10)
			plt.show(block=False)
			title = "ratio of actual change to 1st order predicted change: {0}".format((int)(vectorLength*1000))
			plt.title(title)
			rationame = "/home/alan/Desktop/research/ratio{0}.png".format((int)(vectorLength*1000))
			plt.savefig(rationame)
			plt.close()

			# plot pred vs actual
			plt.figure()
			plt.plot(predChanges)	
			plt.plot(actChanges)	
			plt.show(block=False)
			title = "absolute change in eigenvalue: {0}".format((int)(vectorLength*1000))
			plt.title(title)
			absname = "/home/alan/Desktop/research/abs{0}.png".format((int)(vectorLength*1000))
			plt.savefig(absname)
			plt.close()

def getDecoupledRrtPath(robots, environment, goals):
	obstacleList = environment.getObstacleList()
	graph = robots.getRobotGraph()
	
	rrt_planner = decoupled_rrt.RRT(robot_graph=graph,
			  goal_locs=goals, 
			  obstacle_list=obstacleList,
			  bounds=environment.getBounds())
	# robot_graph, goal_locs, obstacle_list, bounds,
	#              max_move_dist=3.0, goal_sample_rate=5, max_iter=500
	path = rrt_planner.planning()
	return path

def getCoupledAstarPath(robots, environment, goals):
	a_star = coupled_astar.CoupledAstar(robots=robots, env=environment, goals=goals)
	traj = a_star.planning()
	return traj

def main(experiment, swarmInfo, envInfo, seed=999999):
	np.random.seed(seed)

	nRobots, swarmFormation, sensingRadius = swarmInfo
	setting, bounds, nObst, useGrid = envInfo

	envBounds = (0, bounds[0], 0, bounds[1])
	nSquaresWide=bounds[0]
	nSquaresTall=bounds[1]
	nObst = nObst

	# Initialize Robots
	robots = swarm.Swarm(sensingRadius=sensingRadius)
	if swarmFormation=='random':
		robots.initializeSwarm(bounds=bounds, formation=swarmFormation, nRobots=nRobots)
	else:
		robots.initializeSwarm(bounds=bounds, formation=swarmFormation)
	
	# Initialize Environment
	env = environment.Environment(envBounds, useGrid=useGrid, numSquaresWide=nSquaresWide, numSquaresTall=nSquaresTall)
	if setting == 'random':
		env.initializeRandomObstacles(numObstacles=nObst)
		goals = [math_utils.genRandomTuple(lb=0, ub=envBounds[1], size=2) for i in range(robots.getNumRobots())]
	elif setting == 'curve_maze':
		env.initializeCurvesObstacles(numObstacles=nObst)
		mazeGoal = env.getEndOfMazeCenter()
		xlb = mazeGoal[0]-3
		xub = mazeGoal[0]+3
		ylb = mazeGoal[1]-3
		yub = mazeGoal[1]+3
		goals = [math_utils.genRandomLocation(xlb, xub, ylb, yub) for i in range(robots.getNumRobots())]
	else:
		raise NotImplementedError
	
	assert(checkFeasibility(robots, env, goals))

	# if grid convert robot locations to grid indices
	if useGrid:
		startLoc = env.locationListToGridIndexList(robots.getPositionList())
		robots.moveSwarm(startLoc, moveRelative=False)
		goals = env.locationListToGridIndexList(goals)

	# Perform Planning
	if experiment == 'decoupled_rrt': # generate trajectories via naive fully decoupled rrt
		trajs = getDecoupledRrtPath(robots, env, goals) 
	elif experiment == 'coupled_astar':
		trajs = getCoupledAstarPath(robots, env, goals) 
	elif experiment == 'read_file':
		trajs = readTrajFromFile('recent_traj.txt', useGrid=useGrid)
	else:
		raise AssertionError

	trajs = convertAbsoluteTrajToRelativeTraj(trajs)
	testTrajectory(robots, env, trajs, goals, useGrid=useGrid, relativeTraj=True)


if __name__ == '__main__':
	# exp = 'coupled_astar'
	# exp = 'decoupled_rrt'
	exp = 'read_file'

	swarmForm = 'square'
	# swarmForm = 'line'
	# swarmForm = 'rigid_line'
	# swarmForm = 'random'
	nRobots = 30
	sensingRadius = 5

	# setting = 'random'
	setting = 'curve_maze'
	useGrid = False
	envSize = (35, 35)
	numObstacles = 0

	swarmInfo = (nRobots, swarmForm, sensingRadius)
	envInfo = (setting, envSize, numObstacles, useGrid)

	main(experiment=exp, swarmInfo=swarmInfo, envInfo=envInfo)

