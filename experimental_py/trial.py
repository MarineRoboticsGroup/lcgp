import swarm
import rrt_planner

import numpy as np
import matplotlib.pyplot as plt


def makeSensitivityPlotsRandomMotions(robots, environment):
	vectorLength = 0.1

	for vectorLength in [0.001, 0.01, 0.1, 0.15, 0.25, 0.5]:
		robots.initializeSwarm()

		predChanges = []
		actChanges = []
		predRatios = []

		predChangesCrit = []
		actChangesCrit = []
		predRatiosCrit = []

		for i in range(999):

			origEigval = robots.getNthEigval(4)
			grad = robots.getGradientOfNthEigenval(4)
			if grad is False:
				robots.showSwarm()
				break
			else:
				dirVector = math_utils.getRandomVector(len(grad), vectorLength)
				predChange = np.dot(grad, dirVector)
				if origEigval < 1:
					while (predChange < vectorLength*.8):
						dirVector = math_utils.getRandomVector(len(grad), vectorLength)
						predChange = np.dot(grad, dirVector)
				

				robots.moveSwarm(dirVector)
				robots.updateSwarm()
				newEigval = robots.getNthEigval(4)
				actChange = newEigval - origEigval

				predRatio = actChange/predChange

				predChanges.append(predChange)
				actChanges.append(actChange)
				predRatios.append(predRatio)

				if origEigval < 1:
					predChangesCrit.append(predChange)
					actChangesCrit.append(actChange)
					predRatiosCrit.append(predRatio)


		if len(predChanges) > 0:
			# plot ratio
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
			title = "absolute chance in eigenvalue: {0}".format((int)(vectorLength*1000))
			plt.title(title)
			absname = "/home/alan/Desktop/research/abs{0}.png".format((int)(vectorLength*1000))
			plt.savefig(absname)
			plt.close()

def planWithRRT(robots, environment):
    
    goal = [math_utils.genRandomTuple(size=2) for i in range(len(robots))]
    startPos = robots.getPositionList()
    obstacleList = environment.getObstacleList()
    
	# xStartPos = [x[0] for x in startPos]
	# yStartPos = [x[1] for x in startPos]
	# xGoalPos = [x[0] for x in goal]
	# yGoalPos = [x[1] for x in goal]
	# xMin = min(min(xStartPos), min(xGoalPos))
	# yMin = min(min(yStartPos), min(yGoalPos))
	# xMax = max(max(xStartPos), max(xGoalPos))
	# yMax = max(max(yStartPos), max(yGoalPos))

    rrt = rrt_planner.RRT(start=startPos,
              goal=goal,
              rand_area=[-2, 15],
              obstacle_list=obstacleList)
    path = rrt.planning(animation=True)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
        if show_animation:
            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            plt.show()



def main():
	robots = swarm.Swarm(sensingRadius = 10)
	robots.initializeSwarm()



if __name__ == '__main__':
    main()
