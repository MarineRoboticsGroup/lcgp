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

World::World(){}
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
			float range = getRobot(i).distTo(getRobot(j));
			if (range > 0) {
				map.plotCircle(cen, range, j);
			}
		}
	}
}

void World::plotRangeCircles(int id){
	for (int i = 0; i < nRobots; ++i)
	{
		Robot r = getRobot(i);
		Point2d cen = r.getCurrLoc();
		float range = r.distTo(getRobot(id));
		if (range > 0) {
			map.plotCircle(cen, range, id);
		}
	}
}

void World::plotRobotConnections(){
	for (int i = 0; i < nRobots; ++i)
	{
		Point2d pi = getRobot(i).getCurrLoc();
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



void World::fillRanges(){
	float dist;
	for (int i = 0; i < nRobots; ++i)
	{
		Robot& r1 = getRobot(i);
		int id1 = r1.getId();
		for (int j = 0; j < nRobots; ++j)
		{
			Robot& r2 = getRobot(j);
			int id2 = r2.getId();
			if(id1 != id2){
				// remove edge id1->id2
				// g.removeEdge(beacons[id1], beacons[id2]);
				// add new edge
				dist = r1.distTo(r2);
				g.addEdge(beacons[id1], beacons[id2], dist);
			}
		}
	}
}

void World::addRangeMeas(int id1, int id2, float dist){
	g.addEdge(beacons[id1], beacons[id2], dist);
}

// Note: Need to refactor to use graph
void World::printAdjGraph(){
	for (int i = 0; i < nRobots; ++i)
	{
		Robot& r1 = getRobot(i);
		for (int j = 0; j < nRobots; ++j)
		{
			Robot& r2 = getRobot(j);
			printf("%2.1f   ", r1.distTo(r2));;
		}
		std::cout << std::endl;
	}
}


void World::addRobot(Point2d loc){
	vertex_t v = g.addVertex(loc, true);
	beacons.push_back(v);
	nRobots++;
}

void World::addBeacon(Point2d loc){
	vertex_t v = g.addVertex(loc, false);
	beacons.push_back(v);
	nRobots++;
}

/*
 *******CONTROLS**********
*/

void World::randomMovements(){
	float x, y;
	for (int i = 1; i < nRobots; ++i){
		Robot& r = getRobot(i);
		x = genRandom(1);
		y = genRandom(1);
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
			dist = r1.distTo(r2);

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

