import sys
sys.path.insert(1, '/home/alan/range_only_robotics/experimental_py/planners')
import matplotlib.pyplot as plt
import numpy as np
import copy
import flamegraph
import time
import random

# custom libraries
import math_utils
import swarm
import environment
import plot

# planners
import decoupled_rrt
import coupled_astar
import prioritized_prm

def testTrajectory(robots, env, trajs, goals, trajNum=1,
 useGrid=False, delayAnimationStart=False, relativeTraj=False):
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

            robots.moveSwarm(move, moveRelative=relativeTraj)
            robots.updateSwarm()

            graph = robots.getRobotGraph()
            minEigval = robots.getNthEigval(4)
            minEigvals.append(minEigval)
            if useGrid:
                plot.animationWithGrid(graph, env, goals)
            else:
                plot.animationNoGrid(graph, env, goals)
            if minEigval < robots.minEigval:
                print(minEigval, " < ", robots.minEigval)
                plot.plotNthEigenvector(robots, 4)
                plt.pause (5)
            if firstImage:
                plt.pause(10)
                firstImage = False

        plt.close()

        plt.plot(minEigvals)
        plt.hlines([robots.minEigval], 0, len(minEigvals))
        plt.title("Minimum Eigenvalue over Time")
        plt.ylabel("Eigenvalue")
        plt.xlabel("time")
        plt.show()

        with open('recent_traj.txt', 'w') as filehandle:
            for traj in trajs:
                filehandle.write('%s\n' % traj)

def checkFeasibility(swarm, env, goals): # pragma: no cover
    feasible = True
    startEigval = swarm.getNthEigval(4)
    startLocs = swarm.getPositionList()
    graph = swarm.getRobotGraph()
    # plot.plotNoGrid(graph, env, goals)

    if not (env.isFreeSpaceLocListTuples(swarm.getPositionListTuples())):
        print("\nStart Config Inside Obstacles")
        print()
        feasible = False
    if not (env.isFreeSpaceLocListTuples(goals)):
        print("\nGoals Config Inside Obstacles")
        print()
        feasible = False
    goalLoc = []
    for goal in goals:
        goalLoc += list(goal)
    swarm.moveSwarm(goalLoc, moveRelative=False)
    swarm.updateSwarm()
    graph = swarm.getRobotGraph()
    goalEigval = swarm.getNthEigval(4)

    plot.plotNoGrid(graph, env, goals)

    swarm.moveSwarm(startLocs, moveRelative=False)
    swarm.updateSwarm()
    if (startEigval < swarm.minEigval):
        print("\nStarting Config Insufficiently Rigid")
        print("Start Eigenvalue:", startEigval)
        print()
        feasible = False
    if (goalEigval < swarm.minEigval):
        print("\nGoal Config Insufficiently Rigid")
        print("Goal Eigenvalue:", goalEigval)
        print()
        swarm.moveSwarm(goalLoc, moveRelative=False)
        swarm.updateSwarm()
        graph = swarm.getRobotGraph()
        plot.plotNthEigenvector(swarm, 4)
        plot.plotNoGrid(graph, env, goals)
        feasible = False
    return feasible

def readTrajFromFile(filename, useGrid=False):
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

            origEigval = robots.getNthEigval(1)
            grad = robots.getGradientOfNthEigenval(1)
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

def getPriorityPrmPath(robots, environment, goals, useTime):
    priority_prm = prioritized_prm.PriorityPrm(robots=robots, env=environment, goals=goals)
    traj = priority_prm.planning(useTime=useTime)
    return traj

def initGoals(robots):
    goals = [(loc[0]+23, loc[1]+24) for loc in robots.getPositionListTuples()]



    loc1 = (25, 13)
    loc2 = (32, 17)
    loc3 = (27.5, 21.5)
    loc4 = (32, 25)
    loc5 = (27, 26)
    loc6 = (29.5, 29)
    loc7 = (29.5, 32)
    loc8 = (32.5, 33)
    goals = [loc1, loc2, loc3, loc4, loc5, loc6, loc7, loc8]

    # random.shuffle(goals)
    # print("Goals:", goals)
    return goals


def main(experimentInfo, swarmInfo, envInfo, seed=999999):
    np.random.seed(seed)

    expName, useTime, useRelative, showAnimation, profile = experimentInfo
    nRobots, swarmFormation, sensingRadius, minEigval = swarmInfo
    setting, bounds, nObst, useGrid = envInfo

    envBounds = (0, bounds[0], 0, bounds[1])
    nSquaresWide=bounds[0]
    nSquaresTall=bounds[1]
    nObst = nObst


    # Initialize Environment
    env = environment.Environment(envBounds, useGrid=useGrid, numSquaresWide=nSquaresWide, numSquaresTall=nSquaresTall, setting=setting, nObst=nObst)

    # Initialize Robots
    robots = swarm.Swarm(sensingRadius=sensingRadius)
    if swarmFormation=='random':
        robots.initializeSwarm(bounds=bounds, formation=swarmFormation, nRobots=nRobots, minEigval=minEigval)
        while not checkFeasibility(robots, env, goals):
            robots.initializeSwarm(bounds=bounds, formation=swarmFormation, nRobots=nRobots, minEigval=minEigval)
    else:
        robots.initializeSwarm(bounds=bounds, formation=swarmFormation, minEigval=minEigval)

    goals = initGoals(robots)

    assert(checkFeasibility(robots, env, goals))
    assert(nRobots == robots.getNumRobots())
    # if grid convert robot locations to grid indices
    if useGrid:
        startLoc = env.locationListToGridIndexList(robots.getPositionList())
        robots.moveSwarm(startLoc, moveRelative=False)
        goals = env.locationListToGridIndexList(goals)

    # Perform Planning
    startPlanning = time.time()
    if profile:
        flamegraph.start_profile_thread(fd=open("./perf.log", "w"))

    if expName == 'decoupled_rrt': # generate trajectories via naive fully decoupled rrt
        trajs = getDecoupledRrtPath(robots, env, goals)
    elif expName == 'coupled_astar':
        trajs = getCoupledAstarPath(robots, env, goals)
    elif expName == 'priority_prm':
        trajs = getPriorityPrmPath(robots, env, goals, useTime=useTime)
    elif expName == 'read_file':
        trajs = readTrajFromFile('recent_traj.txt', useGrid=useGrid)
    else:
        raise AssertionError

    endPlanning= time.time()
    print('Time Planning:', endPlanning - startPlanning)


    if useRelative:
        print("Converting trajectory from absolute to relative")
        trajs = convertAbsoluteTrajToRelativeTraj(trajs)

    if showAnimation:
        print("Showing trajectory animation")
        testTrajectory(robots, env, trajs, goals, useGrid=useGrid, relativeTraj=useRelative)


if __name__ == '__main__':
    """
    This instantiates and calls everything.
    Any parameters that need to be changed should be accessible from here
    """
    # exp = 'coupled_astar'
    # exp = 'decoupled_rrt'
    exp = 'priority_prm'
    # exp = 'read_file'
    useTime = True
    useRelative = False
    showAnimation = True
    profile = False

    swarmForm = 'square'
    # swarmForm = 'test6'
    swarmForm = 'test8'
    # swarmForm = 'random'
    nRobots = 8
    sensingRadius = 10
    minEigval= 0.1

    # setting = 'random'
    setting = 'curve_maze'
    # setting = 'adversarial1'
    # setting = 'adversarial2'
    useGrid = False
    envSize = (35, 35)
    numObstacles = 0

    experimentInfo = (exp, useTime, useRelative, showAnimation, profile)
    swarmInfo = (nRobots, swarmForm, sensingRadius, minEigval)
    envInfo = (setting, envSize, numObstacles, useGrid)

    main(experimentInfo=experimentInfo, swarmInfo=swarmInfo, envInfo=envInfo)
