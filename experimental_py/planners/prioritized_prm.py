"""

Probablistic Road Map (PRM) Planner

author: Alan Papalia (@alanpapalia)

"""

import random
import math
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import os.path
from os import path
import string
import time
import copy

import math_utils
import plot
import swarm
import kdtree

class PriorityPrm():
    def __init__(self, robots, env, goals):
        # Roadmap Parameters
        self.N_SAMPLE = 3000
        self.N_KNN = 30
        self.MAX_EDGE_LEN = 2
        # swarm
        self.robots = robots
        self.sensingRadius = self.robots.getSensingRadius()
        self.startLocs = self.robots.getPositionListTuples()
        self.numRobots = robots.getNumRobots()
        self.startConfig = self.robots.startConfig
        # environment
        self.env = env
        self.obstacles = env.getObstacleList()
        self.bounds = env.getBounds()
        self.goalLocs = goals
        # Planning Constraints
        self.trajs = None
        self.coordTrajs = None
        # member class objects
        self.roadmap = self.Roadmap(self.robots, self.env, self.goalLocs, self.N_SAMPLE, self.N_KNN, self.MAX_EDGE_LEN)
        self.constraintSets = self.ConstraintSets(self.robots, self.env, self.roadmap)

    def planning(self, useTime=False):
        if not self.performPlanning(useTime):
            raise NotImplementedError
        print("Full set of trajectories found!")
        return self.coordTrajs

    def astarPlanning(self, curRobotId, useTime):
        startId = self.roadmap.getStartIndex(curRobotId)
        goalId = self.roadmap.getGoalIndex(curRobotId)
        print("StartID", startId, self.roadmap.getLocation(startId))
        print("GoalID", goalId, self.roadmap.getLocation(goalId))
        startNode = self.Node(self.startLocs[curRobotId], cost=0.0, pind=-1, timestep=0, index=startId, useTime=useTime)
        goalNode = self.Node(self.goalLocs[curRobotId], cost=0.0, pind=-1, timestep=-1, index=goalId, useTime=useTime)
        openSet, closedSet = dict(), dict()
        openSet[self.getNodeKey(startNode, useTime)] = startNode
        success = False

        while True:
            # if out of options, return conflict information
            if not openSet:
                self.plotFailedSearch(closedSet)
                return ([], success)

            # find minimum cost in openSet
            curKey = min(openSet, key=lambda o: openSet[o].cost + self.calcHeuristic(openSet[o], goalNode, useTime))
            curNode = openSet[curKey]
            # Remove the item from the open set
            del openSet[curKey]
            closedSet[curKey] = curNode
            # If node is valid continue to expand path
            if self.nodeIsValid(curNode, curRobotId):
                # if goal location
                if self.foundGoal(curNode, goalNode):
                    goalNode.pind = curNode.pind
                    goalNode.cost = curNode.cost
                    goalNode.timestep = curNode.timestep
                    goalNode.pkey = curNode.pkey
                    break
                # Add it to the closed set
                closedSet[curKey] = curNode
                curLoc = curNode.getLocation()
                # expand search grid based on motion model
                conns = self.roadmap.getConnections(curNode.index)
                for i in range(len(conns)):
                    new_id = conns[i]
                    if new_id in closedSet:
                        continue
                    newLoc = self.roadmap.getLocation(new_id)
                    dist = self.calcDistanceBetweenLocations(curLoc, newLoc)
                    newNode = self.Node(loc=newLoc, cost=curNode.cost + dist + 1, pind=curNode.index, timestep=curNode.timestep+1, index=new_id, useTime=useTime)
                    newKey = self.getNodeKey(newNode, useTime)
                    # Otherwise if it is already in the open set
                    if newKey in openSet:
                        if openSet[newKey].cost > newNode.cost:
                            # openSet[newKey] = newNode
                            openSet[newKey].cost = newNode.cost
                            openSet[newKey].pind = newNode.pind
                            openSet[newKey].timestep = newNode.timestep
                            openSet[newKey].pkey = newNode.pkey
                    else:
                        openSet[newKey] = newNode
        # generate final course
        pathIdxs = [self.roadmap.getGoalIndex(curRobotId)] # Add goal node
        pkey = goalNode.pkey
        pind = goalNode.pind
        while pind != -1:
            pathIdxs.append(pind)
            node = closedSet[pkey]
            pkey = node.pkey
            pind = node.pind
        pathIdxs.reverse()
        success = True
        return (pathIdxs, success)

    def performPlanning(self, useTime):
        print("Beginning Planning\n")
        self.trajs = [[] for x in range(self.robots.getNumRobots())]
        self.coordTrajs = [[] for x in range(self.robots.getNumRobots())]
        curRobotId = 0
        while curRobotId < self.numRobots:
            print("Planning for robot", curRobotId)
            timeStart = time.time()
            traj, success = self.astarPlanning(curRobotId, useTime)
            timeEnd = time.time()
            print("Planning for robot %d completed in: %f (s)"%(curRobotId, timeEnd-timeStart))
            # Was able to make traj
            if success:
                self.trajs[curRobotId] = traj
                self.coordTrajs[curRobotId] = self.roadmap.convertTrajectoryToCoords(traj)
                self.constraintSets.updateGlobalSetsFromRobotTraj(self.trajs, curRobotId)
                plot.showTrajectories(self.coordTrajs, self.robots, self.env, self.goalLocs)

                # Planning Success Condition
                if curRobotId == self.numRobots-1:
                    print("All Planning Successful")
                    return True
                hasConflict, conflictTime = self.constraintSets.propagateReachability(self.trajs, curRobotId+1)
                if hasConflict:
                    self.constraintSets.animateReachableStates(curRobotId+1)
                    print("Found conflict planning for robot%d \n"%curRobotId)
                    print("Conflict time", conflictTime)
                    conflictLocId = traj[conflictTime]
                    print("Conflict at", conflictLocId, "at time", conflictTime)
                    self.constraintSets.undoGlobalSetsUpdates(curRobotId)
                    self.constraintSets.addConflict(curRobotId, conflictTime, conflictLocId)
                    self.resetTraj(curRobotId)
                else:
                    print("Planning succesful for robot %d \n"%curRobotId)
                    self.constraintSets.clearConflicts(curRobotId+1)
                    curRobotId += 1
            else:
                print("Planning Failed for robot %d. \nReverting to plan for robot %d\n"%(curRobotId, curRobotId-1))
                curRobotId -= 1
                if curRobotId < 0:
                    print("Failed to find paths")
                    return False

    ###### A* Helpers #######
    def nodeIsValid(self, curNode, curRobotId):
        assert(curRobotId >= 0)
        locId = curNode.index
        timestep = curNode.timestep

        return self.stateIsValid(curRobotId, timestep, locId)

    def stateIsValid(self, curRobotId, timestep, locId):
        assert(curRobotId >= 0)
        isReachable = self.constraintSets.isReachableState(curRobotId, timestep, locId)
        if not isReachable:
            return False

        conflictFree = not self.constraintSets.isConflictState(curRobotId, timestep, locId)
        if not conflictFree:
            return False

        if curRobotId == 0:
            isValidState = True
        elif curRobotId == 1:
            curLoc = self.roadmap.getLocation(locId)
            otherLocId = self.constraintSets.getLocIdAtTime(self.trajs, 0, timestep)
            otherLoc = self.roadmap.getLocation(otherLocId)
            if self.calcDistanceBetweenLocations(curLoc, otherLoc) < 2:
                return False
            isValidState = self.constraintSets.isConnectedState(curRobotId, timestep, locId)
        else:
            isValidState = self.constraintSets.isRigidState(curRobotId, timestep, locId)
        return isValidState

    def getNodeKey(self, node, useTime):
        if useTime:
            return (node.index, node.timestep)
        else:
            return node.index

    def foundGoal(self, curNode, goalNode):
        return (curNode.index == goalNode.index)

    def resetTraj(self, curRobotId):
        self.trajs[curRobotId].clear()
        self.coordTrajs[curRobotId].clear()

    ###### General Helpers #######
    def calcHeuristic(self, curNode, goalNode, useTime):
        curLoc = curNode.getLocation()
        goalLoc = goalNode.getLocation()
        nx, ny = curLoc
        gx, gy = goalLoc
        dx = gx-nx
        dy = gy-ny
        if useTime:
            return math.hypot(dx, dy) + 10*curNode.timestep
        else:
            return math.hypot(dx, dy)

    def calcDistanceBetweenLocations(self, loc1, loc2):
        nx, ny = loc1
        gx, gy = loc2
        dx = gx-nx
        dy = gy-ny
        return math.hypot(dx, dy)

    ###### Conversion/Plotting/IO #######
    def plotRoadmap(self):
        print("Displaying Roadmap... May take time :)")
        edges = set()
        for i, _ in enumerate(self.roadmap):
            for ii in range(len(self.roadmap[i])):
                ind = self.roadmap[i][ii]
                edge = (ind, i) if ind < i else (i, ind)
                if edge in edges:
                    continue
                else:
                    edges.add(edge)
                    plt.plot([self.sampleLocs[i][0], self.sampleLocs[ind][0]],
                             [self.sampleLocs[i][1], self.sampleLocs[ind][1]], "-k")

    def plotFailedSearch(self, closedSet):
        nodes = closedSet.values()
        for node in nodes:
            if node.pind == -1:
                continue
            path = [self.roadmap.getLocation(node.index), self.roadmap.getLocation(node.pind)]
            plt.plot(*zip(*path), color='b')
        plot.plotObstacles(self.env)
        plot.plotGoals(self.goalLocs)
        plot.showPlot()

    ###### Member Classes #######
    class Node:
        """
        Node class for dijkstra search
        """

        def __init__(self, loc, cost, pind, timestep, index, useTime):
            self.loc = loc
            self.x = loc[0]
            self.y = loc[1]
            self.cost = cost
            self.pind = pind
            self.timestep = timestep
            self.index = index
            if useTime:
                self.pkey = (pind, timestep-1)
            else:
                self.pkey = pind

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)

        def getLocation(self):
            return self.loc

        def setTimestep(self, timestep):
            self.timestep = timestep

    class Roadmap:
        def __init__(self, robots, env, goalLocs, N_SAMPLE, N_KNN, MAX_EDGE_LEN):
            self.robots = robots
            self.env = env
            self.startLocs = self.robots.getPositionListTuples()
            self.goalLocs = goalLocs
            self.N_SAMPLE = 3000
            self.N_KNN = 30
            self.MAX_EDGE_LEN = 2
            self.roadmapFilename = 'roadmap_%s_%dsamples_%dnn_%dlen_%drob.txt'%(self.robots.startConfig, self.N_SAMPLE, self.N_KNN, self.MAX_EDGE_LEN, self.robots.getNumRobots())
            self.initSampleLocsAndRoadmap()

        def initSampleLocsAndRoadmap(self):
            print("Sampling Locations")
            self.sampleLocs = self.generateSampleLocations()
            self.nodeKDTree = kdtree.KDTree(self.sampleLocs)
            print("%d Locations Sampled\n"%len(self.sampleLocs))
            roadmap = self.readRoadmap()
            if roadmap and (len(roadmap) > 0):
                print("Read from existing roadmap file: %s\n"%self.roadmapFilename)
            else:
                print("%s not found.\nGenerating Roadmap"%self.roadmapFilename)
                roadmap = self.generateRoadmap(self.sampleLocs)
                self.writeRoadmap(roadmap)
                print("New roadmap written to file\n")
            self.roadmap = roadmap

        def generateSampleLocations(self, ):
            xlb, xub, ylb, yub = self.env.bounds
            sampleLocs = []
            while len(sampleLocs) < self.N_SAMPLE:
                newLoc = math_utils.genRandomLocation(xlb, xub, ylb, yub)
                # If not within obstacle
                if self.env.isFreeSpace(newLoc):
                        sampleLocs.append(list(newLoc))
            for loc in self.startLocs:
                    sampleLocs.append(list(loc))
            for loc in self.goalLocs:
                    sampleLocs.append(list(loc))
            return sampleLocs

        def generateRoadmap(self):
            """
            Road map generation
            @return: list of list of edge ids ([[edges, from, 0], ...,[edges, from, N])
            """
            roadmap = []
            nsample = len(self.sampleLocs)
            for curLoc in self.sampleLocs:
                index, dists = self.nodeKDTree.search(np.array(curLoc).reshape(2, 1), k=self.N_KNN)
                inds = index[0]
                edge_id = []
                for ii in range(1, len(inds)):
                    connectingLoc = self.sampleLocs[inds[ii]]
                if self.isValidPath(curLoc, connectingLoc):
                    edge_id.append(inds[ii])
                if len(edge_id) >= self.N_KNN:
                        break
                roadmap.append(edge_id)
            return roadmap

        ###### Accessors #######
        def getConnections(self, locId):
            return self.roadmap[locId]

        def getLocation(self, locId):
            return self.sampleLocs[locId]

        def getNeighborsWithinRadius(self, loc, radius):
            neighborIndexs = self.nodeKDTree.search_in_distance(loc, radius)
            return neighborIndexs

        def getKNearestNeighbors(self, loc, k):
            neighborIndxs, dists = self.nodeKDTree.search(loc, k)
            return neighborIndxs, dists

        def getStartIndex(self, curRobotId):
            index = self.N_SAMPLE + curRobotId
            return index
        def getGoalIndex(self, curRobotId):
            index = self.N_SAMPLE + self.robots.getNumRobots() + curRobotId
            return index
        ###### Conversions #######
        def convertTrajectoriesToCoords(self, trajs):
            newTrajs = []
            for traj in trajs:
                newTraj = []
                for index in traj:
                        newTraj.append(self.sampleLocs[index])
                        newTrajs.append(newTraj)
            return newTrajs

        def convertTrajectoryToCoords(self, traj):
            coords = []
            for index in traj:
                coords.append(self.sampleLocs[index])
            return coords

        ###### Utils #######
        def isValidPath(self, curLoc, connectingLoc):
            dx = curLoc[0] - connectingLoc[0]
            dy = curLoc[1] - connectingLoc[1]
            dist = math.hypot(dx, dy)
            # node too far away
            if dist >= self.MAX_EDGE_LEN:
                    return False
            return self.env.isValidPath(curLoc, connectingLoc)

        def readRoadmap(self,):
            if not path.exists(self.roadmapFilename):
                return False
            rmap = []
            with open(self.roadmapFilename, 'r') as filehandle:
                for line in filehandle:
                        roads = list(map(int, line.split()))
                        rmap.append(roads)
            return rmap

        def writeRoadmap(self):
            with open(self.roadmapFilename, 'w') as filehandle:
                for roads in self.roadmap:
                        line = str(roads).translate(str.maketrans('', '', string.punctuation))
                        filehandle.write('%s\n' % line)

    class ConstraintSets:
        def __init__(self, robots, env, roadmap):
            self.robots = robots
            self.env = env
            self.roadmap = roadmap
            self.numRobots = robots.getNumRobots()
            # Global Sets
            self.connectedStates = [[set()] for x in range(robots.getNumRobots())]    # list of list of sets of tuples
            self.rigidStates = [[set()] for x in range(robots.getNumRobots())]        # list of list of sets of tuples
            # Individual Sets
            self.conflictStates = [[set()] for x in range(robots.getNumRobots())]     # list of list of sets of tuples
            self.reachableStates = [[set()] for x in range(robots.getNumRobots())]    # list of list of sets of tuples
            self.initReachableStates()

        def initReachableStates(self):
            reachSets = [[set([self.roadmap.getStartIndex(robotId)])] for robotId in range(self.numRobots)]
            curRobotId = 0
            startId = self.roadmap.getStartIndex(curRobotId)

            openSet = set()
            connSet = set([startId])

            # continue until robot is capable of reaching all locations
            # Note: Assumes connected roadmap
            while (connSet - openSet):
                openSet.clear()
                openSet.update(connSet)
                connSet.clear()

                # for each node in current state add connected nodes
                for nodeId in openSet:
                    conns = self.roadmap.getConnections(nodeId)
                    connSet.update(conns)

                # clear current state, update feasibility
                reachSets[curRobotId].append(set())
                reachSets[curRobotId][-1].update(connSet)

            self.reachableStates = reachSets
            return True

        def propagateReachability(self, trajs, updateRobotId):
            """
            Should be called while checking trajectory of updateRobotId-1
            Update reachable locs of updateRobot
            """

            startId = self.roadmap.getStartIndex(updateRobotId)
            goalId = self.roadmap.getGoalIndex(updateRobotId)
            self.reachableStates[updateRobotId] = [set([startId])]

            curStep = set([startId])
            nextStep = set(self.roadmap.getConnections(startId))
            timestep = 1
            nextStep = self.getValidSubset(nextStep, trajs, updateRobotId, timestep)

            # continue until robot is capable of reaching all locations
            # Note: Assumes connected roadmap
            while (not (nextStep == curStep)):
                curStep = copy.deepcopy(nextStep)
                for locId in curStep:
                        self.addReachableState(updateRobotId, timestep, locId)
                        nextStep.update(self.roadmap.getConnections(locId))
                        nextStep.add(locId)

                nextStep = self.getValidSubset(nextStep, trajs, updateRobotId, timestep)
                timestep += 1

            if goalId not in curStep:
                # print()
                # print("Conflict Reachable Set", curStep)
                # print()
                hasConflict = True
                failureTime = timestep-1
            else:
                hasConflict = False
                failureTime = None
            return hasConflict, failureTime

        ###### Updates & Undos #######
        def undoGlobalSetsUpdates(self, curRobotId):
            self.connectedStates[curRobotId].clear()
            self.rigidStates[curRobotId].clear()

        def updateGlobalSetsFromRobotTraj(self, trajs, curRobotId):
            assert(trajs[curRobotId])
            for timestep, locId in enumerate(trajs[curRobotId]):
                    self.updateGlobalSetsFromState(trajs, curRobotId, timestep, locId)

        def updateGlobalSetsFromState(self, trajs, curRobotId, curTimestep, locId):
            loc = self.roadmap.getLocation(locId)
            neighbors = self.roadmap.getNeighborsWithinRadius(loc, self.robots.sensingRadius)
            for nodeId in neighbors:
                if self.isConnectedState(curRobotId, curTimestep, nodeId) and curRobotId > 0:
                    if self.isRigidState(curRobotId, curTimestep, nodeId):
                        continue
                    elif self.stateWouldBeRigid(trajs, curRobotId, curTimestep, nodeId):
                        self.addRigidState(curRobotId, curTimestep, nodeId)
                else:
                    self.addConnectedState(curRobotId, curTimestep, nodeId)

        ###### Check Status #######
        def stateWouldBeRigid(self, trajs, curRobotId, curTimestep, nodeId):
            if(curRobotId < 1):
                return False

            locList = [self.roadmap.getLocation(nodeId)]
            for robotId in range(curRobotId+1):
                locId = self.getLocIdAtTime(trajs, robotId, curTimestep)
                loc = self.roadmap.getLocation(locId)
                locList.append(loc)

            isRigid = self.robots.testRigidityFromLocList(locList)
            return isRigid

        def robotHasOptions(self, curRobotId, timestep):
            options = self.reachableStates[curRobotId][timestep] - self.conflictStates[curRobotId][timestep]
            return (len(options) is not 0)

        def isOccupiedState(self, trajs, curRobotId, timestep, locId):
            for robotId in range(curRobotId):
                if self.getLocIdAtTime(trajs, robotId, timestep) == locId:
                    return True
            return False

        def isConnectedState(self, curRobotId, timestep, locId):
            connStates = self.getConnectedStatesAtTime(curRobotId, timestep)
            return locId in connStates

        def isRigidState(self, curRobotId, timestep, locId):
            rigidSet = self.getRigidStatesAtTime(curRobotId, timestep)
            return (locId in rigidSet)

        def isReachableState(self, curRobotId, timestep, locId):
            reachSet = self.getReachableStatesAtTime(curRobotId, timestep)
            return (locId in reachSet)

        def isConflictState(self, curRobotId, timestep, locId):
            if timestep >= len(self.conflictStates[curRobotId]):
                return False
            isConflict = (locId in self.conflictStates[curRobotId][timestep])
            # if isConflict:
            #     print("Conflict found at time %d and location %d"%(timestep, locId))
            return isConflict

        ###### Getters #######
        def getValidSubset(self, origSet, trajs, curRobotId, timestep):
            assert(curRobotId >= 0)
            i = 0
            validSet = set()
            for locId in origSet:
                if self.isOccupiedState(trajs, curRobotId, timestep, locId):
                    continue

                if curRobotId == 0:
                    validSet.add(locId)
                elif curRobotId == 1:
                    if self.isConnectedState(curRobotId, timestep, locId):
                        validSet.add(locId)
                else:
                    if self.isRigidState(curRobotId, timestep, locId):
                        validSet.add(locId)
            return validSet

        def getLocIdAtTime(self, trajs, curRobotId, timestep):
            maxTime = len(trajs[curRobotId])-1
            time = min(maxTime, timestep)
            return trajs[curRobotId][time]
        def getConnectedStatesAtTime(self, curRobotId, timestep):
            connStates = set()
            for robotId in range(curRobotId):
                maxTime = len(self.connectedStates[robotId])-1
                time = min(maxTime, timestep)
                if(time >= 0):
                    conn = self.connectedStates[robotId][time]
                    connStates.update(conn)
            return connStates
        def getRigidStatesAtTime(self, curRobotId, timestep):
            rigStates = set()
            for robotId in range(curRobotId):
                maxTime = len(self.rigidStates[robotId])-1
                time = min(maxTime, timestep)
                if(time >= 0):
                    rigStates.update(self.rigidStates[robotId][time])
            return rigStates

        def getReachableStatesAtTime(self, curRobotId, timestep):
            maxTime = len(self.reachableStates[curRobotId])-1
            time = min(maxTime, timestep)
            return self.reachableStates[curRobotId][time]
        ###### Add/Remove/Clear #######
        def addConnectedState(self, curRobotId, timestep, locId):
            while len(self.connectedStates[curRobotId]) <= timestep:
                self.connectedStates[curRobotId].append(set())
            self.connectedStates[curRobotId][timestep].add(locId)

        def addRigidState(self, curRobotId, timestep, locId):
            assert(self.isConnectedState(curRobotId, timestep, locId))
            while len(self.rigidStates[curRobotId]) <= timestep:
                self.rigidStates[curRobotId].append(set())
            self.rigidStates[curRobotId][timestep].add(locId)

        def addReachableState(self, curRobotId, timestep, locId):
            while(len(self.reachableStates[curRobotId]) <= timestep):
                self.reachableStates[curRobotId].append(set())
            self.reachableStates[curRobotId][timestep].add(locId)

        def addConflict(self, curRobotId, timestep, locId):
            while len(self.conflictStates[curRobotId]) <= timestep:
                self.conflictStates[curRobotId].append(set())
            self.conflictStates[curRobotId][timestep].add(locId)

        def removeConnectedState(self, curRobotId, timestep, locId):
            assert(self.isConnectedState(curRobotId, timestep, locId))
            self.connectedStates[curRobotId][timestep].remove(locId)

        def removeRigidState(self, curRobotId, timestep, locId):
            assert(self.isRigidState(curRobotId, timestep, locId))
            self.rigidStates[curRobotId][timestep].remove(locId)

        def clearConflicts(self, curRobotId):
            self.conflictStates[curRobotId].clear()

        def clearReachableStates(self, curRobotId):
            self.reachableStates[curRobotId].clear()

        ###### Plotting #######
        def animateConnectedStates(self, curRobotId):
            print("Plotting Connected States")
            trajLens = [len(x) for x in self.connectedStates]
            maxTimestep = max(trajLens)
            plt.close()
            for timestep in range(maxTimestep):
                plot.clearPlot()
                plt.title("Connected States: Robot %d timestep %d"%(curRobotId, timestep))
                self.plotConnectedStates(curRobotId+1, timestep)
                plot.plotObstacles(self.env)
                plot.setXlim(self.env.getBounds()[0], self.env.getBounds()[1])
                plot.setYlim(self.env.getBounds()[2], self.env.getBounds()[3])
                plot.showPlotAnimation()

        def animateRigidStates(self, curRobotId):
            print("Plotting Rigid States")
            trajLens = [len(x) for x in self.connectedStates]
            maxTimestep = max(trajLens)
            plt.close()
            for timestep in range(maxTimestep):
                plot.clearPlot()
                plt.title("Rigid States: Robot %d timestep %d"%(curRobotId, timestep))
                self.plotRigidStates(curRobotId+1, timestep)
                plot.plotObstacles(self.env)
                plot.setXlim(self.env.getBounds()[0], self.env.getBounds()[1])
                plot.setYlim(self.env.getBounds()[2], self.env.getBounds()[3])
                plot.showPlotAnimation()

        def animateReachableStates(self, curRobotId):
            print("Plotting Reachable States")
            maxTimestep = len(self.reachableStates[curRobotId])
            plt.close()
            for timestep in range(maxTimestep):
                plot.clearPlot()
                plt.title("Reachable States: Robot %d timestep %d"%(curRobotId, timestep))
                self.plotConnectedStates(curRobotId, timestep)
                self.plotRigidStates(curRobotId, timestep)
                self.plotReachableStates(curRobotId, timestep)
                plt.legend(["Connected", "Rigid", "Reachable"])
                self.plotEnv()
                plot.showPlotAnimation()

        def plotConnectedStates(self, curRobotId, timestep):
            locIds = self.getConnectedStatesAtTime(curRobotId, timestep)
            pts = []
            for locId in locIds:
                pts.append(self.roadmap.getLocation(locId))
            xLocs = [x[0] for x in pts]
            yLocs = [x[1] for x in pts]
            plt.scatter(xLocs, yLocs, color='g')

        def plotRigidStates(self, curRobotId, timestep):
            locIds = self.getRigidStatesAtTime(curRobotId, timestep)
            pts = []
            for locId in locIds:
                pts.append(self.roadmap.getLocation(locId))
            xLocs = [x[0] for x in pts]
            yLocs = [x[1] for x in pts]
            plt.scatter(xLocs, yLocs, color='y')

        def plotReachableStates(self, curRobotId, timestep):
            locIds = self.getReachableStatesAtTime(curRobotId, timestep)
            pts = []
            for locId in locIds:
                pts.append(self.roadmap.getLocation(locId))
            xLocs = [x[0] for x in pts]
            yLocs = [x[1] for x in pts]
            plt.scatter(xLocs, yLocs, color='b')

        def plotLocIdList(self, locIds):
            pts = []
            for locId in locIds:
                pts.append(self.roadmap.getLocation(locId))
            xLocs = [x[0] for x in pts]
            yLocs = [x[1] for x in pts]
            plt.scatter(xLocs, yLocs)

        def plotEnv(self):
            plot.plotObstacles(self.env)
            plot.setXlim(self.env.getBounds()[0], self.env.getBounds()[1])
            plot.setYlim(self.env.getBounds()[2], self.env.getBounds()[3])

