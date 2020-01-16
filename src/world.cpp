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

void World::plotEstLocs(){
	for (int i = 0; i < nRobots; ++i)
	{
		map.plotPoint(estLocs[i], i, 0.5);
	}
}

void World::plotRangeCircles(){
	for (int i = 0; i < nRobots; ++i)
	{
		vertex_t beac1 = beacons[i];
		Point2d cen = getRobot(i).getCurrLoc();
		for (int j = 0; j < nRobots; ++j)
		{
			vertex_t beac2 = beacons[j];
			float range = g.getEdge(beac1, beac2);
			if (range > 0) {
				map.plotCircle(cen, range, j);
			}
		}
	}
}

void World::plotRangeCircles(int id){
	vertex_t beac1 = beacons[id];
	for (int i = 0; i < nRobots; ++i)
	{
		vertex_t beac2 = beacons[i];
		Point2d cen = getRobot(i).getCurrLoc();
		float range = g.getEdge(beac1, beac2);
		if (range > 0) {
			map.plotCircle(cen, range, id);
		}
	}
}

void World::plotRangeCirclesEst(){
	for (int i = 0; i < nRobots; ++i)
	{
		vertex_t beac1 = beacons[i];
		Point2d cen = estLocs[i];
		for (int j = 0; j < nRobots; ++j)
		{
			vertex_t beac2 = beacons[j];
			float range = g.getEdge(beac1, beac2);
			if (range > 0) {
				map.plotCircle(cen, range, j);
			}
		}
	}
}

void World::plotRangeCirclesEst(int id){
	vertex_t beac1 = beacons[id];
	for (int i = 0; i < nRobots; ++i)
	{
		vertex_t beac2 = beacons[i];
		Point2d cen = estLocs[i];
		float range = g.getEdge(beac1, beac2);
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

void World::setAxis(){
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

void World::adjustLims(){
	float xmax = 0, ymax = 0;
	float x, y;
	for (auto pts : estLocs){
		x = abs(pts.getX());
		y = abs(pts.getY());
		if (x > xmax)
		{
			xmax = x;
		}
		if (y > ymax)
		{
			ymax = y;
		}
	}
	for (int i = 0; i < nRobots; ++i){
		x = abs(g.getVertex(i).getCurrLoc().getX());
		y = abs(g.getVertex(i).getCurrLoc().getY());
		if (x > xmax)
		{
			xmax = x;
		}
		if (y > ymax)
		{
			ymax = y;
		}
	}
	if (xlim - xmax > 10 || xlim - xmax < 1) { xlim = xmax + 5;}
	if (ylim - ymax > 10 || ylim - ymax < 1) { ylim = ymax + 5;}

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
		for (int j = i; j < nRobots; ++j)
		{
			Robot& r2 = getRobot(j);
			int id2 = r2.getId();
			if(id1 != id2){
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

void World::realizeGraph(){
	g.realizeGraphSCIP(estLocs);
}


void World::addRobot(Point2d loc){
	vertex_t v = g.addVertex(loc, true);
	beacons.push_back(v);
	estLocs.push_back(loc);
	nRobots++;
}


void World::addRobot(Point2d loc, float stdDev){
	vertex_t v = g.addVertex(loc, true, stdDev);
	beacons.push_back(v);
	estLocs.push_back(loc);
	nRobots++;
}


void World::addBeacon(Point2d loc){
	vertex_t v = g.addVertex(loc, false);
	beacons.push_back(v);
	estLocs.push_back(loc);
	nRobots++;
}

void World::addBeacon(Point2d loc, float stdDev){
	vertex_t v = g.addVertex(loc, false);
	beacons.push_back(v);
	estLocs.push_back(loc);
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

