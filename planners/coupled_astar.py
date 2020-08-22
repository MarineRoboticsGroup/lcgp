import copy
import string
import math
from itertools import permutations
from os import path
import itertools
import chaospy
import numpy as np
import matplotlib.pyplot as plt

import graph
import kdtree

class CoupledAstar():
    def __init__(self, robots, env, goals, min_eigval=0.75):
        self.robots = robots
        self.env = env
        self.goalLocs = goals
        self.startLocs = self.robots.get_position_list_tuples()
        self.min_eigval = min_eigval

        self.numRobots = robots.get_num_robots()

        roadmap_sampling = "uniform"
        self.N_SAMPLE = 200
        self.N_KNN = 4
        self.MAX_EDGE_LEN = 8
        self.roadmap = self.Roadmap(self.robots, self.env, self.goalLocs, roadmap_sampling, self.N_SAMPLE, self.N_KNN, self.MAX_EDGE_LEN)
        self.plotRoadmap()
        self.startIndexs = self.roadmap.getStartIndexList()
        self.goalIndexs = self.roadmap.getGoalIndexList()
        print("Start Index:", self.startIndexs)
        print("Goal Index:", self.goalIndexs)

    class CoupledAstarNode:
        def __init__(self, indexList, cost, pind):
            self.indexList = indexList
            self.cost = cost
            self.pind = pind

        def getNodeHash(self):
            for i, indexs in enumerate(self.indexList):
                if i is 0:
                    h = indexs
                else:
                    h += indexs
            return h

        def getIndexList(self):
            return self.indexList

    def planning(self,):
        nstart = self.CoupledAstarNode(self.startIndexs, 0.0, -1)
        ngoal = self.CoupledAstarNode(self.goalIndexs, 0.0, -1)

        goal_id = ngoal.getNodeHash()
        open_set, closed_set = dict(), dict()
        open_set[nstart.getNodeHash()] = nstart

        cnt = 0
        while 1:
            cnt += 1
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(open_set, key=lambda o: open_set[o].cost + self.calcHeuristic(open_set[o], ngoal))
            curNode = open_set[c_id]

            if self.isGoal(curNode):
                print("Find goal")
                ngoal.pind = curNode.pind
                ngoal.cost = curNode.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = curNode

            # expand_grid search grid based on motion model
            curLocs = curNode.getIndexList()
            connLocs = [self.roadmap.getConnections(locId) for locId in curLocs]
            nextSteps = [s for s in itertools.product(*connLocs)]

            coordList = [self.roadmap.getLocation(loc_ind) for loc_ind in curLocs]
            print("Current Locs:", coordList)
            print("Iteration:", cnt)
            print("Cost:", curNode.cost)
            print()

            for indexList in nextSteps:
                # make new node
                distFromCurNode = self.sumOfDistancesBetweenIndexLists(curLocs, indexList)
                node = self.CoupledAstarNode(list(indexList), curNode.cost + distFromCurNode, c_id)
                n_id = node.getNodeHash()

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        return self.calcFinalPath(ngoal, closed_set)

    def calcFinalPath(self, ngoal, closed_set):
        locPath = [[] for i in range(self.robots.get_num_robots())]

        locs = ngoal.getIndexList()
        for i, loc in enumerate(locs):
            locPath[i].append(loc)

        pind = ngoal.pind
        while pind != -1:
            n = closed_set[pind]
            locs = n.getIndexList()
            for i, loc in enumerate(locs):
                locPath[i].append(loc)
            pind = n.pind
        for path in locPath:
            path.reverse()
        return locPath

    def calcHeuristic(self, curNode, goalNode):
        curIndexList = curNode.getIndexList()
        goalIndexList = goalNode.getIndexList()
        return self.sumOfDistancesBetweenIndexLists(curIndexList, goalIndexList)

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

    def verify_node(self, node):
        locList = node.getIndexList()
        # check if inside bounds
        for i, loc in enumerate(locList):
            # check for collisions
            for otherloc in locList[i+1:]:
                if loc == otherloc:
                    return False
        # check geometry
        coordList = [self.roadmap.getLocation(loc_ind) for loc_ind in locList]
        g = graph.Graph()
        g.initialize_from_location_list(coordList, self.robots.get_sensing_radius())
        eigval = g.get_nth_eigval(4)
        # if eigval < self.min_eigval:
        #     print("\nInvalid Configuration\n")
        #     return False

        return True

    def sumOfDistancesBetweenIndexLists(self, locationList, goalList):
        assert(len(locationList) == len(goalList))
        dists = []
        for i, _ in enumerate(locationList):
            dists.append(self.distanceBetweenLocs(locationList[i], goalList[i]))
        return sum(dists)

    def distanceBetweenLocs(self, loc1, loc2):
        x1, y1 = self.roadmap.getLocation(loc1)
        x2, y2 = self.roadmap.getLocation(loc2)
        deltax = x1-x2
        deltay = y1-y2
        return math.hypot(deltax, deltay)

    def isGoal(self, node):
        return (node.getIndexList() == self.goalIndexs)

    class Roadmap:
        def __init__(self, robots, env, goalLocs, sampling_type, N_SAMPLE, N_KNN, MAX_EDGE_LEN):
            self.robots = robots
            self.env = env
            self.startLocs = self.robots.get_position_list_tuples()
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
            for loc in self.startLocs:
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
        def getStartIndexList(self):
            indexList = []
            for i in range(self.robots.get_num_robots()):
                indexList.append(self.getStartIndex(i))
            return indexList
        def getGoalIndexList(self):
            indexList = []
            for i in range(self.robots.get_num_robots()):
                indexList.append(self.getGoalIndex(i))
            return indexList
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

