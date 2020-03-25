import swarm

import numpy as np
import matplotlib.pyplot as plt



def makeSensitivityPlots():
	sensorRadius = 10
	robots = swarm.Swarm(sensingRadius = sensorRadius)
	robots.initializeSwarm()
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


