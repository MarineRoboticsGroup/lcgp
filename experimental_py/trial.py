import swarm

import numpy as np
import matplotlib.pyplot as plt


# https://pypi.org/project/qpsolvers/
# def calcEqualEigvalContour(gradient):
# 	perpVectors = []
# 	vec = np.cross(gradient, gradient)
# 	print(vec)
# 	perpVectors.append(vec)
	
# 	for i in range(len(gradient)-2):
# 		vec = np.cross(gradient, perpVectors[i])
# 		print(vec)
# 		perpVectors.append(vec)

def getRandomUnitVector(nDim, length):
	vec = np.random.uniform(low=-2, high=2, size=nDim)
	vec = vec/np.linalg.norm(vec,2)
	vec *= length
	return vec

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

		for i in range(50):
			# robots.printAllEigvals()
			# print()
			# print()
			# robots.showSwarm()


			origEigval = robots.getNthEigval(4)
			grad = robots.getGradientOfNthEigenval(4)
			if grad is False:
				robots.showSwarm()
				break
			else:
				dirVector = getRandomUnitVector(len(grad), vectorLength)
				predChange = np.dot(grad, dirVector)
				if origEigval < 1:
					while (predChange < vectorLength*.8):
						dirVector = getRandomUnitVector(len(grad), vectorLength)
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

				if abs(predRatio) > 20:
					print("Eigenvalue", origEigval)
					print("Predicted", predChange)
					print("Actual", actChange)
					print("Pred Ratio", predRatio)
					print()
					print("---------------------")
					print()

		if len(predChanges) > 0:
			if len(predChangesCrit) > 0:
				print(predChangesCrit)
				print(actChangesCrit)
				print(np.corrcoef(predChangesCrit, actChangesCrit))
				print()
				# plt.plot(predRatiosCrit)
			print(np.corrcoef(actChanges, predChanges))
			print()
			print()

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


