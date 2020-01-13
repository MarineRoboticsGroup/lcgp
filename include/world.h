#include "geom.h"
#include "plot.h"
#include "robot.h"
#include "graph.h"

#include <vector>
#include <cstdio>


class World
{
private:
	Plot map;
	int nRobots = 0;
	DistanceGraph g;
	std::vector <vertex_t> beacons;

public:
	World();
	~World();

		// Plotting controls
	void plotRangeCircles();
	void plotRangeCircles(int id);
	void plotRobots();
	void plotRobotConnections();
	void setAxis(float xlim, float ylim);
	void setAxisEqual();
	void showMap(std::string display);

		// Map Accessors
	void addRobot(Point2d loc);
	void addRobot(Point2d loc, float stdDev);
	void addBeacon(Point2d loc);
	Robot& getRobot(int id) {return g.getVertex(id);}

		// Graph Control
	void fillRanges();
	void addRangeMeas(int id1, int id2, float dist);
	void printGraphInfo() { g.printInfo(); }
	void printAdjGraph();
	void printGraphReal() { 
		// g.realizeGraphIPOPT(); 
		g.realizeGraphSCIP(); 
	}

		// Controls
	void randomMovements();
	bool robustQuadMaxDist();
	bool localizeRobot(Robot r);
	void localizeAll();

};