#include "geom.h"
#include "plot.h"
#include "robot.h"
#include "graph.h"

#include <vector>
#include <cstdio>


class World
{
	public:
		World(int maxNRobots);
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
		bool addRobot(Robot r);
		Robot& getRobot(int id) {return robots[id];}
		Point2d getEstLoc(int id) const {return estLocs[id];}

		// Graph Control
		void runGraphSample();
		void fillRanges();
		void addEdge(int id1, int id2, float dist) {edges[id1][id2] = dist;}
		void printAdjGraph();

		// Controls
		void randomMovements();
		bool robustQuadMaxDist();
		bool localizeRobot(Robot r);
		void localizeAll();

	private:
		Plot map;

		int nRobots = 0;

		DistanceGraph g;

		std::vector <Point2d> estLocs;
		std::vector <Robot> robots;
		std::vector<std::vector<float>> edges;
};