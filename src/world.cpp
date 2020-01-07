#include "world.h"


namespace {
	
	float genRandom(float bound){
		// std::srand(std::time(nullptr));
		float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		r -= .5;
		r *= bound;
		return r;
	}



}

World::World(int maxNRobots){
	// initialize 2d vector for robot connections
	for (int i = 0; i < maxNRobots; ++i)
	{
		edges.push_back(std::vector<float>(maxNRobots));
	}
}
World::~World(){}


void World::plotRobots(){
	for (int i = 0; i < nRobots; ++i)
	{
		map.plotPoint(getRobot(i).getCurrLoc(), i);
	}
}

void World::plotRangeCircles(){
	for (int i = 0; i < nRobots; ++i)
	{
		Point2d cen = getRobot(i).getCurrLoc();
		for (int j = 0; j < nRobots; ++j)
		{
			float range = getRobot(i).getRange(j);
			if (range > 0) {
				map.plotCircle(cen, range, j);
			}
		}
	}
}

void World::plotRangeCircles(int id){
	for (int i = 0; i < nRobots; ++i)
	{
		Point2d cen = getRobot(i).getCurrLoc();
		float range = getRobot(i).getRange(id);
		if (range > 0) {
			map.plotCircle(cen, range, id);
		}
	}
}

void World::plotRobotConnections(){
	for (int i = 0; i < nRobots; ++i)
	{
		Point2d pi = robots[i].getCurrLoc();
		for (int j = i+1; j < nRobots; ++j)
		{
			Point2d pj = getRobot(j).getCurrLoc();
			map.plotLine(pi, pj);
		}
	}
}

void World::setAxis(float xlim, float ylim){
	map.setAxis(xlim, ylim);
}

void World::setAxisEqual(){
	map.setAxisEqual();
}

void World::showMap(std::string display){
	if (display == "animation"){
		map.animation();
	} else if (display == "static"){
		map.showPlot();
	}
}



/*
 *******GRAPH CONTROLS**********
*/

void World::runGraphSample(){
	g.runSample();
}

void World::fillRanges(){
	for (int i = 0; i < nRobots; ++i)
	{
		Robot& r1 = getRobot(i);
		int id1 = r1.getRobotId();
		for (int j = 0; j < nRobots; ++j)
		{
			Robot& r2 = getRobot(j);
			int id2 = r2.getRobotId();
			if(id1 != id2){
				edges[id1][id2] = r1.distToRob(r2);
			}
		}
	}
}

void World::printAdjGraph(){
	for (int i = 0; i < nRobots; ++i)
	{
		for (int j = 0; j < nRobots; ++j)
		{
			printf("%.1f   ", edges[i][j]);;
		}
		std::cout << std::endl;
	}
}

bool World::addRobot(Robot r){
	int rId = r.getRobotId();
	if (rId < nRobots){
		return false;
	} 
	robots.push_back(r);
	nRobots++;
	return true;
}


/*
 *******CONTROLS**********
*/

void World::randomMovements(){
	float x, y;
	for (int i = 1; i < nRobots; ++i){
		Robot& r = getRobot(i);
		x = genRandom(.1);
		y = genRandom(.1);
		r.move(x, y);
	}
}



// TODO
bool World::robustQuadMaxDist(){
	float x, y, dist, maxDist;
	for (int i = 0; i < nRobots; ++i){
		Robot& r1 = getRobot(i);
		maxDist = 0.0;
		for (int j = 0; j < nRobots; ++j){
			Robot& r2 = getRobot(j);
			dist = r1.distToRob(r2);

			// y = genRandom(.1);
			// r1.move(x, y);
		}
	}
	return false;
}


// TODO
bool World::localizeRobot(Robot r){
	return false;
}


// TODO
void World::localizeAll(){
	for (int i = 0; i < nRobots; ++i){
		Robot& r = getRobot(i);
		World::localizeRobot(r);
	}
}

