import graph

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

colors = ['b','g','r','c','m','y']


####### Animation Calls #######

def animationNoGrid(graph, env, goals):
	clearPlot()
	plotGraphWithEdges(graph)
	plotObstacles(env)
	plotGoals(goals)
	setXlim(env.getBounds()[0], env.getBounds()[1])
	setYlim(env.getBounds()[2], env.getBounds()[3])
	# showPlot()
	showPlotAnimation()

def animationWithGrid(graph, env, goals):
	goalList = env.gridIndexListToLocationList(goals)
	grid = env.getGrid()
	clearPlot()
	plotGraphWithEdges(graph)
	plotObstacles(env)
	plotGoals(goalList)
	setXlim(env.getBounds()[0], env.getBounds()[1])
	setYlim(env.getBounds()[2], env.getBounds()[3])
	plotGrid(grid)
	# showPlot()
	showPlotAnimation()

####### Single Frame Calls #######

def plotNoGrid(graph, env, goals):
	clearPlot()
	plotGraphWithEdges(graph)
	plotObstacles(env)
	plotGoals(goals)
	setXlim(env.getBounds()[0], env.getBounds()[1])
	setYlim(env.getBounds()[2], env.getBounds()[3])
	showPlot()

def plotNoGridNoGoals(graph, env):
	clearPlot()
	plotGraphWithEdges(graph)
	plotObstacles(env)
	setXlim(env.getBounds()[0], env.getBounds()[1])
	setYlim(env.getBounds()[2], env.getBounds()[3])
	showPlot()

def plotWithGrid(graph, env, goals):
	goalList = env.gridIndexListToLocationList(goals)
	grid = env.getGrid()
	clearPlot()
	plotGraphWithEdges(graph)
	plotObstacles(env)
	plotGoals(goalList)
	setXlim(env.getBounds()[0], env.getBounds()[1])
	setYlim(env.getBounds()[2], env.getBounds()[3])
	plotGrid(grid)
	showPlot()

def showTrajectories(trajs, robots, env, goals):
	for i, traj in enumerate(trajs):
		if traj == []:
			break
		plt.plot(*zip(*traj), color=colors[i%6])
	plotGraphNoEdges(robots.getRobotGraph())
	plotObstacles(env)
	plotGoals(goals)
	showPlot()
	plt.close()


####### Atomic Calls #######

def plotGraphWithEdges(graph):
	nodeLocations = graph.getNodeLocationList()
	nodeXLocs = [x[0] for x in nodeLocations]
	nodeYLocs = [x[1] for x in nodeLocations]
	for i, nodeLoc in enumerate(nodeLocations):
		plt.scatter(nodeLoc[0], nodeLoc[1], color=colors[i%6])
	
	plotEdges(graph)

def plotGraphNoEdges(graph):
	nodeLocations = graph.getNodeLocationList()
	edges = graph.getGraphEdgeList()

	for i, nodeLoc in enumerate(nodeLocations):
		plt.scatter(nodeLoc[0], nodeLoc[1], color=colors[i%6])


def plotGoals(goals):
	for i, goalLoc in enumerate(goals):
		plt.scatter(goalLoc[0], goalLoc[1], color=colors[i%6], marker='x')

def plotEdges(graph):
	nodeLocations = graph.getNodeLocationList()
	edges = graph.getGraphEdgeList()
	nodeXLocs = [x[0] for x in nodeLocations]
	nodeYLocs = [x[1] for x in nodeLocations]
	for e in edges:
		xs = [nodeXLocs[e[0]], nodeXLocs[e[1]]]
		ys = [nodeYLocs[e[0]], nodeYLocs[e[1]]]
		plt.plot(xs, ys, color='k')

def plotObstacles(env):
	obstacles = env.getObstacleList()
	fig = plt.gcf()
	ax = fig.gca()
	for obs in obstacles:
		circ = plt.Circle(obs.getCenter(), obs.getRadius(), color='r')
		ax.add_artist(circ)

def plotGrid(grid):
	fig = plt.gcf()
	ax = fig.gca()

	nRow = len(grid)
	nCol = len(grid[0])

	squareWidth, squareHeight = grid[0][0].getGridSquareSize()
	xoff = squareWidth/2
	yoff = squareHeight/2
	gridColors = np.ones((nRow, nCol)) * np.nan
	for row in range(nRow):
		ax.axhline(row, lw=0.1, color='k', zorder=5)

		for col in range(nCol): 
			if (not grid[row][col].isSquareFree()):
				gridColors[row,col] = 0

			if row is 0:
				rightBorder = grid[row][col].getGridSquareCenter()[0] + (squareWidth/2)
				ax.axvline(col, lw=0.1, color='k', zorder=5)

	my_cmap = matplotlib.colors.ListedColormap(['r'])
	# set the 'bad' values (nan) to be white and transparent
	my_cmap.set_bad(color='w', alpha=0)

	# draw the boxes
	ax.imshow(gridColors, interpolation='none', cmap=my_cmap, extent=[0, nRow, 0, nRow], zorder=0)


####### Basic Controls #######

def setXlim(lb, ub):
	plt.xlim(lb, ub)

def setYlim(lb, ub):
	plt.ylim(lb, ub)

def showPlot():
	plt.show(block=True)

def showPlotAnimation():
	plt.pause(0.1)
	plt.show(block=False)


def clearPlot():
	plt.clf()