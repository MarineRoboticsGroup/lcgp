import sys
sys.path.insert(1, '/home/alan/range_only_robotics/experimental_py/planners')


import math_utils
import swarm
import environment
import plot

# planners
import decoupled_rrt
import coupled_astar

import matplotlib.pyplot as plt
import numpy as np

def readTrajFromFile(filename):
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
				xcoord = float(sect[:commaInd])
				ycoord = float(sect[commaInd+1:])
				coords = (xcoord, ycoord)
				traj.append(coords)

				line = line[endInd+1:]
				startInd = line.find('(')
				endInd = line.find(')')

			trajs.append(traj)
	return trajs

def testTrajectory(robots, env, trajs, goals, trajNum=1, useGrid=False, delayAnimationStart=False):
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
		trajIndex = [0 for traj in trajs]
		finalTrajIndex = [len(traj) for traj in trajs]
		newPos = []
		minEigvals = []

		while not (trajIndex == finalTrajIndex):
			graph = robots.getRobotGraph()
			minEigvals.append(robots.getNthEigval(4))
			
			if firstImage:
				plt.pause(10)
				firstImage = False

			newPos.clear()
			for robotIndex in range(robots.getNumRobots()):
				newLoc = trajs[robotIndex][trajIndex[robotIndex]]
				
				if useGrid:
					newLoc = env.gridIndexToLocation(newLoc)
				
				newPos += list(newLoc)

				if trajIndex[robotIndex] != finalTrajIndex[robotIndex]:
					trajIndex[robotIndex] += 1

			robots.moveSwarm(newPos, moveRelative=False)
			robots.updateSwarm()

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

		with open('recent_traj.txt', 'w') as filehandle:
			for traj in trajs:
				filehandle.write('%s\n' % traj)

				

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

def main(experiment='coupled_astar', seed=100, useGrid=False, nObst=0, bounds=(10,10)):
	np.random.seed(seed)
	envBounds = (0, bounds[0], 0, bounds[1])
	nSquaresWide=bounds[0]
	nSquaresTall=bounds[1]
	nObst = nObst

	robots = swarm.Swarm(sensingRadius = 30)
	env = environment.Environment(envBounds, useGrid=useGrid, numSquaresWide=nSquaresWide, numSquaresTall=nSquaresTall)
	
	robots.initializeSwarm()
	env.initializeRandom(numObstacles=nObst)
	goals = [math_utils.genRandomTuple(lb=0, ub=envBounds[1], size=2) for i in range(robots.getNumRobots())]

	if useGrid:
		robotLoc = env.locationListToGridIndexList(robots.getPositionList())
		startLoc = []
		for loc in robotLoc:
			startLoc.append(loc[0])
			startLoc.append(loc[1])

		robots.moveSwarm(startLoc, moveRelative=False)
		goals = env.locationListToGridIndexList(goals)

	if experiment == 'decoupled_rrt': # generate trajectories via naive fully decoupled rrt
		trajs = getDecoupledRrtPath(robots, env, goals) 
	elif experiment == 'coupled_astar':
		trajs = getCoupledAstarPath(robots, env, goals) 
	elif experiment == 'read_file':
		trajs = readTrajFromFile('recent_traj.txt')
	else:
		raise AssertionError

	testTrajectory(robots, env, trajs, goals, useGrid=useGrid)


if __name__ == '__main__':
	exp = 'coupled_astar'
	# exp = 'decoupled_rrt'
	# exp = 'read_file'

	useGrid = True
	envSize = (3, 3)
	numObstacles = 0

	main(experiment=exp, useGrid=useGrid, bounds=envSize, nObst=numObstacles)

