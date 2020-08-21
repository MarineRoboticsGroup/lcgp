import math_utils
import swarm
import environment
import plot

import time
import numpy as np
import matplotlib.pyplot as plt

colors = ['b','g','r','c','m','y']
nEig = 4

def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()
def ClickPlaceNodes(noise_model, noise_stddev):
    sensingRadius = 0.5
    robots = swarm.Swarm(sensingRadius, noise_model, noise_stddev)
    envBounds = (0, 1, 0, 1)
    env = environment.Environment(envBounds, useGrid=False, numSquaresWide=1, numSquaresTall=1, setting='empty', nObst=0)

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
        plot.plotNoGridNoGoalsNoBlock(graph, env)
        if robots.getNumRobots() >= 3:
            eigval = robots.getNthEigval(nEig)
            print(eigval)
            plot.plotNthEigenvector(robots,nEig)

def PrintEigenvalOfLocs(loc_list, noise_model, noise_stddev):
    sensingRadius = 100
    robots = swarm.Swarm(sensingRadius, noise_model, noise_stddev)
    min_x = min(loc_list[:,0]) - 1
    max_x = max(loc_list[:,0]) + 1
    min_y = min(loc_list[:,1]) - 1
    max_y = max(loc_list[:,1]) + 1
    envBounds = (min_x, max_x, min_y, max_y)
    env = environment.Environment(envBounds, useGrid=False, numSquaresWide=1, numSquaresTall=1, setting='empty', nObst=0)
    robots.initializeSwarmFromLocationListTuples(loc_list)
    graph = robots.getRobotGraph()
    if robots.getNumRobots() >= 3:
        eigval = robots.getNthEigval(nEig)
        print(eigval)
    else:
        print("Needs more nodes")
    # plot.plotNoGridNoGoalsNoBlock(graph, env)
    # plot.plotNthEigenvector(robots,nEig)
    # plt.show(block=True)
    return eigval

def rotMat(thetaDegrees):
    angle = thetaDegrees
    theta = (angle/180.) * np.pi
    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]])
    return rotMatrix

def testCircleSwitch():
    loc_list = []
    eigvals = []
    theta = np.linspace(0, np.pi, num=100)
    for i in theta:
        delX = 2*np.cos(i)
        delY = 2*np.sin(i)
        locs1 = np.array([(3+delX, 3+delY), (3-delX, 3-delY), (1, 4)])
        loc_list.append(locs1)
        eigvals.append(PrintEigenvalOfLocs(locs1))

    eigvals = np.array(eigvals)
    loc_list = np.array(loc_list)
    for i, _ in enumerate(theta):
        plt.figure(1)
        plt.clf()
        plt.xlim(0, 5)
        plt.ylim(0, 5)
        for colnum, locs in enumerate(loc_list[i]):
            plt.scatter(locs[0], locs[1], color=colors[colnum%6])
        plt.pause(0.1)
        plt.show(block=False)

        plt.figure(2)
        plt.clf()
        plt.plot(eigvals[:i])
        plt.pause(0.1)
        plt.show(block=False)

    plt.show(block=True)



# testCircleSwitch()
ClickPlaceNodes()
