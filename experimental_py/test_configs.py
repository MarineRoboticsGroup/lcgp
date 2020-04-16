import math_utils
import swarm
import environment
import plot

import time
import numpy as np
import matplotlib.pyplot as plt


def tellme(s):
	print(s)
	plt.title(s, fontsize=16)
	plt.draw()


# swarm
sensingRadius = .25
robots = swarm.Swarm(sensingRadius=sensingRadius)

# environment
envBounds = (0, 1, 0, 1)
env = environment.Environment(envBounds, useGrid=False, numSquaresWide=1, numSquaresTall=1)
# env.initializeRandomObstacles(numObstacles=0)



tellme('Click where you would like to place a node')

plt.waitforbuttonpress()

pts = []
even = True
while True:
	if even:
		tellme('Click for new node')
		temp = np.asarray(plt.ginput(1, timeout=-1))
		pts.append(tuple(temp[0]))
	else:
		tellme('Keypress to quit')
		if plt.waitforbuttonpress():
				break

	even = not even
	robots.initializeSwarmFromLocationListTuples(pts)
	graph = robots.getRobotGraph()
	plot.plotNoGridNoGoals(graph, env)

	if robots.getNumRobots() >= 3:
		eigval = robots.getNthEigval(4)
		print(eigval)
