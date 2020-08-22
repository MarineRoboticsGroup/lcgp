"""

Probablistic Road Map (PRM) Planner

author: Alan Papalia (@alanpapalia)

"""

import random
import math
import numpy as np
import chaospy
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
colors = ['b','g','r','c','m','y']

class PriorityPrm():
    def __init__(self, robots, env, goals):
        # Roadmap Parameters
        self.N_SAMPLE = 1000
        self.N_KNN = 10
        self.MAX_EDGE_LEN = 2
        # swarm
        self.robots = robots
        self.sensingRadius = self.robots.get_sensing_radius()
        self.start_loc_list = self.robots.get_position_list_tuples()
        self.numRobots = robots.get_num_robots()
        self.startConfig = self.robots.startConfig
        # environment
        self.env = env
        self.obstacles = env.get_obstacle_list()
        self.bounds = env.get_bounds()
        self.goalLocs = goals
        # Planning Constraints
        self.trajs = None
        self.coordTrajs = None
        # member class objects
        # roadmap_sampling = "random"
        roadmap_sampling = "uniform"
        self.roadmap = self.Roadmap(self.robots, self.env, self.goalLocs, roadmap_sampling, self.N_SAMPLE, self.N_KNN, self.MAX_EDGE_LEN)
        # self.plotRoadmap()
        self.constraintSets = self.ConstraintSets(self.robots, self.env, self.roadmap)

    def planning(self, useTime=False):
        if not self.performPlanning(useTime):
            raise NotImplementedError
        print("Full set of trajectories found!")
        return self.coordTrajs

    def astarPlanning(self, cur_robot_id, useTime):
        startId = self.roadmap.getStartIndex(cur_robot_id)
        goalId = self.roadmap.getGoalIndex(cur_robot_id)
        print("StartID", startId, self.roadmap.getLocation(startId))
        print("GoalID", goalId, self.roadmap.getLocation(goalId))
        startNode = self.Node(self.start_loc_list[cur_robot_id], cost=0.0, pind=-1, timestep=0, index=startId, useTime=useTime)
        goalNode = self.Node(self.goalLocs[cur_robot_id], cost=0.0, pind=-1, timestep=-1, index=goalId, useTime=useTime)
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
            if self.nodeIsValid(curNode, cur_robot_id):
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
                    # newNode = self.Node(loc=newLoc, cost=curNode.cost + dist + 1, pind=curNode.index, timestep=curNode.timestep+1, index=new_id, useTime=useTime)
                    newNode = self.Node(loc=newLoc, cost=curNode.cost + 1, pind=curNode.index, timestep=curNode.timestep+1, index=new_id, useTime=useTime)
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
        pathIdxs = [self.roadmap.getGoalIndex(cur_robot_id)] # Add goal node
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
        self.trajs = [[] for x in range(self.robots.get_num_robots())]
        self.coordTrajs = [[] for x in range(self.robots.get_num_robots())]
        cur_robot_id = 0
        while cur_robot_id < self.numRobots:
            print("Planning for robot", cur_robot_id)
            timeStart = time.time()
            traj, success = self.astarPlanning(cur_robot_id, useTime)
            timeEnd = time.time()
            print("Planning for robot %d completed in: %f (s)"%(cur_robot_id, timeEnd-timeStart))
            # Was able to make traj
            if success:
                self.trajs[cur_robot_id] = traj
                self.coordTrajs[cur_robot_id] = self.roadmap.convertTrajectoryToCoords(traj)
                self.constraintSets.updateGlobalSetsFromRobotTraj(self.trajs, cur_robot_id)
                # plot.plot_trajectories(self.coordTrajs, self.robots, self.env, self.goalLocs)

                # Planning Success Condition
                if cur_robot_id == self.numRobots-1:
                    print("All Planning Successful")
                    print()
                    return True
                hasConflict, conflictTime = self.constraintSets.propagateValidity(self.trajs, cur_robot_id+1)
                if hasConflict:
                    print("Found conflict planning for robot%d at time %d "%(cur_robot_id, conflictTime))
                    self.constraintSets.animateValidStates(self.coordTrajs, cur_robot_id+1)
                    mintime = min(conflictTime, len(traj)-1)
                    # conflictLocId = traj[conflictTime]
                    conflictLocId = traj[mintime]
                    print("Conflict at", conflictLocId, "at time", conflictTime, "\n")
                    self.constraintSets.undoGlobalSetsUpdates(cur_robot_id)
                    self.constraintSets.addConflict(cur_robot_id, conflictTime, conflictLocId)
                    self.resetTraj(cur_robot_id)
                else:
                    print("Planning succesful for robot %d \n"%cur_robot_id)
                    # if cur_robot_id == 0:
                    #     self.constraintSets.animateConnectedStates(cur_robot_id+1, self.coordTrajs, self.goalLocs)
                    # else:
                    #     self.constraintSets.animateRigidStates(cur_robot_id+1, self.coordTrajs, self.goalLocs)
                    self.constraintSets.clearConflicts(cur_robot_id+1)
                    cur_robot_id += 1
            else:
                print("Planning Failed for robot %d. \nReverting to plan for robot %d\n"%(cur_robot_id, cur_robot_id-1))
                cur_robot_id -= 1
                if cur_robot_id < 0:
                    print("Failed to find paths")
                    return False

    ###### A* Helpers #######
    def nodeIsValid(self, curNode, cur_robot_id):
        assert(cur_robot_id >= 0)
        loc_id = curNode.index
        timestep = curNode.timestep
        return self.stateIsValid(cur_robot_id, timestep, loc_id)

    def stateIsValid(self, cur_robot_id, timestep, loc_id):
        assert(cur_robot_id >= 0)
        isReachable = self.constraintSets.isReachableState(cur_robot_id, timestep, loc_id)
        if not isReachable:
            return False
        conflictFree = not self.constraintSets.isConflictState(cur_robot_id, timestep, loc_id)
        if not conflictFree:
            print("Has Conflict, denying robot %d, time: %d, loc: %d"%(cur_robot_id, timestep, loc_id))
            return False
        isValidState = self.constraintSets.isValidState(cur_robot_id, timestep, loc_id)
        if not isValidState:
            return False
        if cur_robot_id == 1 :
            loc0 = self.getLocAtTime(0, timestep)
            loc1 = self.roadmap.getLocation(loc_id)
            if self.calcDistanceBetweenLocations(loc0, loc1) < 1:
                return False
        # if cur_robot_id == 2:
        #     loc0 = self.getLocAtTime(0, timestep)
        #     loc1 = self.getLocAtTime(1, timestep)
        #     loc2 = self.roadmap.getLocation(loc_id)
        #     if self.calcDistanceBetweenLocations(loc0, loc2) < 2:
        #         return False
        #     if self.calcDistanceBetweenLocations(loc1, loc2) < 1:
        #         return False
        return True

    def getNodeKey(self, node, useTime):
        if useTime:
            return (node.index, node.timestep)
        else:
            return node.index

    def foundGoal(self, curNode, goalNode):
        return (curNode.index == goalNode.index)

    def resetTraj(self, cur_robot_id):
        self.trajs[cur_robot_id].clear()
        self.coordTrajs[cur_robot_id].clear()

    ###### General Helpers #######
    def calcHeuristic(self, curNode, goalNode, useTime):
        curLoc = curNode.getLocation()
        goalLoc = goalNode.getLocation()
        nx, ny = curLoc
        gx, gy = goalLoc
        dx = gx-nx
        dy = gy-ny
        if useTime:
            return 0.5 * math.hypot(dx, dy)
        else:
            return math.hypot(dx, dy)

    def calcDistanceBetweenLocations(self, loc1, loc2):
        nx, ny = loc1
        gx, gy = loc2
        dx = gx-nx
        dy = gy-ny
        return math.hypot(dx, dy)

    def getLocIdAtTime(self, cur_robot_id, timestep):
        maxTime = len(self.trajs[cur_robot_id])-1
        time = min(maxTime, timestep)
        return self.trajs[cur_robot_id][time]
    def getLocAtTime(self, cur_robot_id, timestep):
        maxTime = len(self.trajs[cur_robot_id])-1
        time = min(maxTime, timestep)
        return self.roadmap.getLocation(self.trajs[cur_robot_id][time])
    ###### Conversion/Plotting/IO #######
    def plotRoadmap(self):
        print("Displaying Roadmap... May take time :)")
        edges = set()
        for i, _ in enumerate(self.roadmap.roadmap):
            for ii in range(len(self.roadmap.roadmap[i])):
                ind = self.roadmap.roadmap[i][ii]
                edge = (ind, i) if ind < i else (i, ind)
                if edge in edges:
                    continue
                else:
                    edges.add(edge)
                    plt.plot([self.roadmap.sampleLocs[i][0], self.roadmap.sampleLocs[ind][0]],
                             [self.roadmap.sampleLocs[i][1], self.roadmap.sampleLocs[ind][1]], "-k")
        plt.show(block=True)

    def plotFailedSearch(self, closedSet):
        nodes = closedSet.values()
        print("Plotting the Failed Search!!")
        plot.clear_plot()
        plt.close()
        for node in nodes:
            if node.pind == -1:
                continue
            path = [self.roadmap.getLocation(node.index), self.roadmap.getLocation(node.pind)]
            plt.plot(*zip(*path), color='b')
        plot.plot_obstacles(self.env)
        plot.plot_goals(self.goalLocs)
        plot.showPlot()
        plt.close()

    ###### Member Classes #######
    class Node:
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
        def __init__(self, robots, env, goalLocs, sampling_type, N_SAMPLE, N_KNN, MAX_EDGE_LEN):
            self.robots = robots
            self.env = env
            self.start_loc_list = self.robots.get_position_list_tuples()
            self.goalLocs = goalLocs
            self.sampling_type = sampling_type
            self.N_SAMPLE = N_SAMPLE
            self.N_KNN = N_KNN
            self.MAX_EDGE_LEN = MAX_EDGE_LEN
            self.roadmapFilename = 'roadmap_%s_%s_%s_%dsamples_%dnn_%dlen_%drob.txt'%(self.env.setting, self.sampling_type, self.robots.startConfig, self.N_SAMPLE, self.N_KNN, self.MAX_EDGE_LEN, self.robots.get_num_robots())
            self.initSampleLocsAndRoadmap()

        def initSampleLocsAndRoadmap(self):
            print("Building Roadmap")
            if self.sampling_type == "random":
                self.sampleLocs = np.array(self.generateSampleLocationsRandom())
            elif self.sampling_type == "uniform":
                self.sampleLocs = self.generateSampleLocationsUniform()
            else:
                raise NotImplementedError
            self.nodeKDTree = kdtree.KDTree(self.sampleLocs)
            roadmap = self.readRoadmap()
            if roadmap and (len(roadmap) > 0):
                print("Read from existing roadmap file: %s\n"%self.roadmapFilename)
                self.roadmap = roadmap
            else:
                print("%s not found.\nGenerating Roadmap"%self.roadmapFilename)
                self.roadmap = self.generateRoadmap()
                self.writeRoadmap()
                print("New roadmap written to file\n")

        def generateSampleLocationsRandom(self, ):
            xlb, xub, ylb, yub = self.env.bounds
            sampleLocs = []
            while len(sampleLocs) < self.N_SAMPLE:
                newLoc = math_utils.generate_random_loc(xlb, xub, ylb, yub)
                # If not within obstacle
                if self.env.is_free_space(newLoc):
                        sampleLocs.append(list(newLoc))
            for loc in self.start_loc_list:
                    sampleLocs.append(list(loc))
            for loc in self.goalLocs:
                    sampleLocs.append(list(loc))
            return sampleLocs

        def generateSampleLocationsUniform(self, ):
            xlb, xub, ylb, yub = self.env.bounds
            sampleLocs = []
            distribution = chaospy.J(chaospy.Uniform(xlb, xub), chaospy.Uniform(ylb, yub))
            samples = distribution.sample(self.N_SAMPLE*10, rule="halton")
            i = 0
            while len(sampleLocs) < self.N_SAMPLE and i < len(samples[0]):
                newLoc = samples[:, i]
                i += 1
                # If not within obstacle
                if self.env.is_free_space(newLoc):
                        sampleLocs.append(list(newLoc))
            if len(sampleLocs) < self.N_SAMPLE:
                print("Not able to fully build roadmap. Need more samples")
                raise NotImplementedError
            for loc in self.start_loc_list:
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
                # print(inds)
                edge_id = []
                for ii in range(1, len(inds)):
                    connectingLoc = self.sampleLocs[inds[ii]]
                    if self.is_valid_path(curLoc, connectingLoc):
                        edge_id.append(inds[ii])
                        if len(edge_id) >= self.N_KNN:
                            break
                roadmap.append(edge_id)
            return roadmap

        ###### Accessors #######
        def getConnections(self, loc_id):
            return self.roadmap[loc_id]

        def getLocation(self, loc_id):
            return self.sampleLocs[loc_id]

        def getNeighborsWithinRadius(self, loc, radius):
            neighborIndexs = self.nodeKDTree.search_in_distance(loc, radius)
            return neighborIndexs

        def getKNearestNeighbors(self, loc, k):
            neighborIndxs, dists = self.nodeKDTree.search(loc, k)
            return neighborIndxs, dists

        def getStartIndex(self, cur_robot_id):
            index = self.N_SAMPLE + cur_robot_id
            return index
        def getGoalIndex(self, cur_robot_id):
            index = self.N_SAMPLE + self.robots.get_num_robots() + cur_robot_id
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
                coords.append(tuple(self.sampleLocs[index]))
            return coords

        ###### Utils #######
        def is_valid_path(self, curLoc, connectingLoc):
            dx = curLoc[0] - connectingLoc[0]
            dy = curLoc[1] - connectingLoc[1]
            dist = math.hypot(dx, dy)
            # node too far away
            if dist >= self.MAX_EDGE_LEN:
                    return False
            return self.env.is_valid_path(curLoc, connectingLoc)

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
            self.numRobots = robots.get_num_robots()
            # Global Sets
            self.connected_states = [[set()] for x in range(robots.get_num_robots())]    # list of list of sets of tuples
            self.rigid_states = [[set()] for x in range(robots.get_num_robots())]        # list of list of sets of tuples
            # Individual Sets
            self.conflictStates = [[set()] for x in range(robots.get_num_robots())]     # list of list of sets of tuples
            self.reachable_states = [[set()] for x in range(robots.get_num_robots())]    # list of list of sets of tuples
            self.valid_states = [[set()] for x in range(robots.get_num_robots())]    # list of list of sets of tuples
            self.initReachableAndValidStates()

        def initReachableAndValidStates(self):
            reachSets = [[set([self.roadmap.getStartIndex(robotId)])] for robotId in range(self.numRobots)]
            valid_set = [[set([self.roadmap.getStartIndex(robotId)])] for robotId in range(self.numRobots)]
            for cur_robot_id in range(self.robots.get_num_robots()):
                startId = self.roadmap.getStartIndex(cur_robot_id)

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
                    reachSets[cur_robot_id].append(set())
                    reachSets[cur_robot_id][-1].update(connSet)
                    if cur_robot_id == 0:
                        valid_set[cur_robot_id].append(set())
                        valid_set[cur_robot_id][-1].update(connSet)

            self.reachable_states = reachSets
            self.valid_states = valid_set

        def propagateValidity(self, trajs, update_robot_id):
            """
            Should be called while checking trajectory of update_robot_id-1
            Update valid locs of updateRobot
            """
            startId = self.roadmap.getStartIndex(update_robot_id)
            goalId = self.roadmap.getGoalIndex(update_robot_id)
            self.valid_states[update_robot_id] = [set([startId])]
            timestep = 0
            curStep = set([startId])
            curStep = self.getValidSubset(curStep, trajs, update_robot_id, timestep)
            timestep += 1
            nextStep = set(self.roadmap.getConnections(startId))
            nextStep = self.getValidSubset(nextStep, trajs, update_robot_id, timestep)
            # continue until robot is capable of reaching all locations
            # Note: Assumes connected roadmap
            cnt = 2
            while (cnt and len(nextStep) > 0 and goalId not in curStep):
                curStep = copy.deepcopy(nextStep)
                for loc_id in curStep:
                    nextStep.update(self.roadmap.getConnections(loc_id))
                    nextStep.add(loc_id)
                nextStep = self.getValidSubset(nextStep, trajs, update_robot_id, timestep)
                for loc_id in nextStep:
                    self.addValidState(update_robot_id, timestep, loc_id)
                timestep += 1
                if not nextStep - curStep:
                    cnt -= 1

            if goalId not in curStep:
                hasConflict = True
                failureTime = timestep
            else:
                hasConflict = False
                failureTime = None
            return hasConflict, failureTime

        ###### Updates & Undos #######
        def undoGlobalSetsUpdates(self, cur_robot_id):
            self.connected_states[cur_robot_id].clear()
            self.rigid_states[cur_robot_id].clear()

        def updateGlobalSetsFromRobotTraj(self, trajs, cur_robot_id):
            assert(trajs[cur_robot_id])
            for timestep, loc_id in enumerate(trajs[cur_robot_id]):
                    self.updateGlobalSetsFromState(trajs, cur_robot_id, timestep, loc_id)

        def updateGlobalSetsFromState(self, trajs, cur_robot_id, curTimestep, loc_id):
            loc = self.roadmap.getLocation(loc_id)
            neighbors = self.roadmap.getNeighborsWithinRadius(loc, self.robots.sensingRadius)
            for nodeId in neighbors:
                if self.isOccupiedState(trajs, cur_robot_id+1, curTimestep, nodeId):
                    continue
                elif self.isConnectedState(cur_robot_id, curTimestep, nodeId) and cur_robot_id > 0:
                    if self.isRigidState(cur_robot_id, curTimestep, nodeId):
                        continue
                    elif self.stateWouldBeRigid(trajs, cur_robot_id, curTimestep, nodeId):
                        self.addRigidState(cur_robot_id, curTimestep, nodeId)
                else:
                    self.addConnectedState(cur_robot_id, curTimestep, nodeId)

        ###### Check Status #######
        def stateWouldBeRigid(self, trajs, cur_robot_id, curTimestep, nodeId):
            if(cur_robot_id < 1):
                return False
            locList = [self.roadmap.getLocation(nodeId)]
            for robotId in range(cur_robot_id+1):
                loc_id = self.getLocIdAtTime(trajs, robotId, curTimestep)
                loc = self.roadmap.getLocation(loc_id)
                locList.append(loc)
            isRigid = self.robots.test_rigidity_from_loc_list(locList)
            return isRigid
        def robotHasOptions(self, cur_robot_id, timestep):
            options = self.reachable_states[cur_robot_id][timestep] - self.conflictStates[cur_robot_id][timestep]
            return (len(options) is not 0)

        def isOccupiedState(self, trajs, cur_robot_id, timestep, loc_id):
            for robotId in range(cur_robot_id):
                if self.getLocIdAtTime(trajs, robotId, timestep) == loc_id:
                    return True
            return False

        def isConnectedState(self, cur_robot_id, timestep, loc_id):
            connStates = self.getConnectedStatesAtTime(cur_robot_id, timestep)
            return loc_id in connStates

        def isRigidState(self, cur_robot_id, timestep, loc_id):
            rigidSet = self.getRigidStatesAtTime(cur_robot_id, timestep)
            return (loc_id in rigidSet)

        def isValidState(self, cur_robot_id, timestep, loc_id):
            valid_set = self.getValidStatesAtTime(cur_robot_id, timestep)
            return (loc_id in valid_set)

        def isReachableState(self, cur_robot_id, timestep, loc_id):
            reachSet = self.getReachableStatesAtTime(cur_robot_id, timestep)
            return (loc_id in reachSet)

        def isConflictState(self, cur_robot_id, timestep, loc_id):
            if timestep >= len(self.conflictStates[cur_robot_id]):
                return False
            isConflict = (loc_id in self.conflictStates[cur_robot_id][timestep])
            # if isConflict:
            #     print("Conflict found at time %d and location %d"%(timestep, loc_id))
            return isConflict

        ###### Getters #######
        # This is where what is valid is determined
        def getValidSubset(self, origSet, trajs, cur_robot_id, timestep):
            assert(cur_robot_id >= 0)
            i = 0
            validSet = set()
            for loc_id in origSet:
                if self.isOccupiedState(trajs, cur_robot_id, timestep, loc_id):
                    continue
                if cur_robot_id == 0:
                    validSet.add(loc_id)
                # elif cur_robot_id == 1:
                elif cur_robot_id <= 3 :
                    if self.isConnectedState(cur_robot_id, timestep, loc_id):
                        validSet.add(loc_id)
                else:
                    if self.isRigidState(cur_robot_id, timestep, loc_id):
                        validSet.add(loc_id)
            return validSet

        def getLocIdAtTime(self, trajs, cur_robot_id, timestep):
            maxTime = len(trajs[cur_robot_id])-1
            time = min(maxTime, timestep)
            return trajs[cur_robot_id][time]
        def getLocAtTime(self, trajs, cur_robot_id, timestep):
            maxTime = len(trajs[cur_robot_id])-1
            time = min(maxTime, timestep)
            return self.roadmap.getLocation(trajs[cur_robot_id][time])
        def getConnectedStatesAtTime(self, cur_robot_id, timestep):
            connStates = set()
            for robotId in range(cur_robot_id):
                maxTime = len(self.connected_states[robotId])-1
                time = min(maxTime, timestep)
                if(time >= 0):
                    conn = self.connected_states[robotId][time]
                    connStates.update(conn)
            return connStates
        def getRigidStatesAtTime(self, cur_robot_id, timestep):
            rigStates = set()
            for robotId in range(cur_robot_id):
                maxTime = len(self.rigid_states[robotId])-1
                time = min(maxTime, timestep)
                if(time >= 0):
                    rigStates.update(self.rigid_states[robotId][time])
            return rigStates

        def getReachableStatesAtTime(self, cur_robot_id, timestep):
            maxTime = len(self.reachable_states[cur_robot_id])-1
            time = min(maxTime, timestep)
            return self.reachable_states[cur_robot_id][time]
        def getValidStatesAtTime(self, cur_robot_id, timestep):
            maxTime = len(self.valid_states[cur_robot_id])-1
            time = min(maxTime, timestep)
            return self.valid_states[cur_robot_id][time]
        ###### Add/Remove/Clear #######
        def addConnectedState(self, cur_robot_id, timestep, loc_id):
            while len(self.connected_states[cur_robot_id]) <= timestep:
                self.connected_states[cur_robot_id].append(set())
            self.connected_states[cur_robot_id][timestep].add(loc_id)

        def addRigidState(self, cur_robot_id, timestep, loc_id):
            assert(self.isConnectedState(cur_robot_id, timestep, loc_id))
            while len(self.rigid_states[cur_robot_id]) <= timestep:
                self.rigid_states[cur_robot_id].append(set())
            self.rigid_states[cur_robot_id][timestep].add(loc_id)

        def addReachableState(self, cur_robot_id, timestep, loc_id):
            while(len(self.reachable_states[cur_robot_id]) <= timestep):
                self.reachable_states[cur_robot_id].append(set())
            self.reachable_states[cur_robot_id][timestep].add(loc_id)

        def addValidState(self, cur_robot_id, timestep, loc_id):
            while(len(self.valid_states[cur_robot_id]) <= timestep):
                self.valid_states[cur_robot_id].append(set())
            self.valid_states[cur_robot_id][timestep].add(loc_id)

        def addConflict(self, cur_robot_id, timestep, loc_id):
            while len(self.conflictStates[cur_robot_id]) <= timestep:
                self.conflictStates[cur_robot_id].append(set())
            self.conflictStates[cur_robot_id][timestep].add(loc_id)

        def removeConnectedState(self, cur_robot_id, timestep, loc_id):
            assert(self.isConnectedState(cur_robot_id, timestep, loc_id))
            self.connected_states[cur_robot_id][timestep].remove(loc_id)

        def removeRigidState(self, cur_robot_id, timestep, loc_id):
            assert(self.isRigidState(cur_robot_id, timestep, loc_id))
            self.rigid_states[cur_robot_id][timestep].remove(loc_id)

        def clearConflicts(self, cur_robot_id):
            self.conflictStates[cur_robot_id].clear()

        def clearReachableStates(self, cur_robot_id):
            self.reachable_states[cur_robot_id].clear()

        def clearValidStates(self, cur_robot_id):
            self.validStates[cur_robot_id].clear()

        ###### Plotting #######
        def animateConnectedStates(self, cur_robot_id, coordTrajs, goalLocs):
            # plot.plot_trajectories(self.coordTrajs, self.robots, self.env, self.goalLocs)
            print("Plotting Connected States")
            trajLens = [len(x) for x in self.connected_states]
            maxTimestep = max(trajLens)
            plt.close()
            for timestep in range(maxTimestep):
                plot.clear_plot()
                plt.title("Connected States: Robot %d timestep %d"%(cur_robot_id, timestep))
                self.plotConnectedStates(cur_robot_id+1, timestep)
                for i, traj in enumerate(coordTrajs):
                    if traj == []:
                        continue
                    time = min(timestep, trajLens[i]-1)
                    loc = traj[time]
                    plt.scatter(loc[0], loc[1], color=colors[i%6])
                    plt.plot(*zip(*traj), color=colors[i%6])

                plot.plot_obstacles(self.env)
                plot.set_x_lim(self.env.get_bounds()[0], self.env.get_bounds()[1])
                plot.set_y_lim(self.env.get_bounds()[2], self.env.get_bounds()[3])
                plot.showPlotAnimation()
                # if timestep == 0:
                #     plt.pause(10)
            plt.close()

        def animateRigidStates(self, cur_robot_id, coordTrajs, goalLocs):
            print("Plotting Rigid States")
            trajLens = [len(x) for x in self.connected_states]
            maxTimestep = max(trajLens)
            plt.close()
            # plt.pause(5)
            for timestep in range(maxTimestep):
                plot.clear_plot()
                plt.title("Rigid States: Robot %d timestep %d"%(cur_robot_id, timestep))
                self.plotRigidStates(cur_robot_id+1, timestep)
                for i, traj in enumerate(coordTrajs):
                    if traj == []:
                        continue
                    time = min(timestep, trajLens[i]-1)
                    loc = traj[time]
                    plt.scatter(loc[0], loc[1], color=colors[i%6])
                    plt.plot(*zip(*traj), color=colors[i%6])

                plot.plot_obstacles(self.env)
                plot.set_x_lim(self.env.get_bounds()[0], self.env.get_bounds()[1])
                plot.set_y_lim(self.env.get_bounds()[2], self.env.get_bounds()[3])
                plot.showPlotAnimation()
                # if timestep == 0:
                #     plt.pause(10)
            plt.close()

        def animateReachableStates(self, cur_robot_id):
            print("Plotting Reachable States")
            maxTimestep = len(self.reachable_states[cur_robot_id])
            plt.close()
            for timestep in range(maxTimestep):
                plot.clear_plot()
                plt.title("Reachable States: Robot %d timestep %d"%(cur_robot_id, timestep))
                self.plotConnectedStates(cur_robot_id, timestep)
                self.plotRigidStates(cur_robot_id, timestep)
                self.plotReachableStates(cur_robot_id, timestep)
                plt.legend(["Connected", "Rigid", "Reachable"])
                self.plotEnv()
                plot.showPlotAnimation()

        def animateValidStates(self, trajs, cur_robot_id):
            print("Plotting Valid States")
            goalId = self.roadmap.getGoalIndex(cur_robot_id)
            goalLoc = self.roadmap.getLocation(goalId)
            maxTimestep = len(self.valid_states[cur_robot_id])
            plt.close()
            trajLen = [len(traj) for traj in trajs]
            for timestep in range(maxTimestep):
                plot.clear_plot()
                plt.title("Valid States: Robot %d timestep %d"%(cur_robot_id, timestep))
                self.plotConnectedStates(cur_robot_id, timestep)
                self.plotRigidStates(cur_robot_id, timestep)
                self.plotValidStates(cur_robot_id, timestep)
                plt.scatter(goalLoc[0], goalLoc[1], color='k')
                # for i, traj in enumerate(trajs):
                #     if traj == []:
                #         continue
                #     time = min(timestep, trajLen[i]-1)
                #     loc = traj[time]
                #     # loc = self.roadmap.getLocation(traj[time])
                #     plt.scatter(loc[0], loc[1], color=colors[i%6])
                #     plt.plot(*zip(*traj), color=colors[i%6])
                plt.legend(["Connected", "Rigid", "Valid", "GOAL"])
                self.plotEnv()
                plot.showPlotAnimation()

        def plotConnectedStates(self, cur_robot_id, timestep):
            loc_ids = self.getConnectedStatesAtTime(cur_robot_id, timestep)
            pts = []
            for loc_id in loc_ids:
                pts.append(self.roadmap.getLocation(loc_id))
            xLocs = [x[0] for x in pts]
            yLocs = [x[1] for x in pts]
            plt.scatter(xLocs, yLocs, color='g')

        def plotRigidStates(self, cur_robot_id, timestep):
            loc_ids = self.getRigidStatesAtTime(cur_robot_id, timestep)
            pts = []
            for loc_id in loc_ids:
                pts.append(self.roadmap.getLocation(loc_id))
            xLocs = [x[0] for x in pts]
            yLocs = [x[1] for x in pts]
            plt.scatter(xLocs, yLocs, color='y')

        def plotReachableStates(self, cur_robot_id, timestep):
            loc_ids = self.getReachableStatesAtTime(cur_robot_id, timestep)
            pts = []
            for loc_id in loc_ids:
                pts.append(self.roadmap.getLocation(loc_id))
            xLocs = [x[0] for x in pts]
            yLocs = [x[1] for x in pts]
            plt.scatter(xLocs, yLocs, color='b')

        def plotValidStates(self, cur_robot_id, timestep):
            loc_ids = self.getValidStatesAtTime(cur_robot_id, timestep)
            pts = []
            for loc_id in loc_ids:
                pts.append(self.roadmap.getLocation(loc_id))
            xLocs = [x[0] for x in pts]
            yLocs = [x[1] for x in pts]
            plt.scatter(xLocs, yLocs, color='c')

        def plotLocIdList(self, loc_ids):
            pts = []
            for loc_id in loc_ids:
                pts.append(self.roadmap.getLocation(loc_id))
            xLocs = [x[0] for x in pts]
            yLocs = [x[1] for x in pts]
            plt.scatter(xLocs, yLocs)

        def plotEnv(self):
            plot.plot_obstacles(self.env)
            plot.set_x_lim(self.env.get_bounds()[0], self.env.get_bounds()[1])
            plot.set_y_lim(self.env.get_bounds()[2], self.env.get_bounds()[3])

