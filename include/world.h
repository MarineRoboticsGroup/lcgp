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
	std::vector <Point2d> estLocs;
	float xlim = 10, ylim = 10;

public:
	World();
	~World();

		// Plotting controls
	void plotRangeCircles();
	void plotRangeCircles(int id);
	void plotRangeCirclesEst();
	void plotRangeCirclesEst(int id);
	void plotRobots();
	void plotEstLocs();
	void plotRobotConnections();
	void setAxis();
	void setAxisEqual();
	void showMap(std::string display);
	void adjustLims();

		// Map Accessors
	void addRobot(Point2d loc);
	void addRobot(Point2d loc, float stdDev);
	void addBeacon(Point2d loc);
	void addBeacon(Point2d loc, float stdDev);
	Robot& getRobot(int id) {return g.getVertex(id);}

		// Graph Control
	void fillRanges();
	void addRangeMeas(int id1, int id2);
	void printGraphInfo() { g.printInfo(); }
	void printAdjGraph();
	void realizeGraph();

		// Controls
	void randomMovements();
	bool robustQuadMaxDist();
	bool localizeRobot(Robot r);
	void localizeAll();

};