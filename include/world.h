#include "geom.h"
#include "plot.h"
#include "robot.h"
#include "graph.h"

#include <vector>
#include <cstdio>


class World
{
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
		bool addRobot(Robot r);
		void addRobot(Point2d loc);
		Robot& getRobot(int id) {return g.getVertex(id);}
		Point2d getEstLoc(int id) const {return estLocs[id];}

		// Graph Control
		void fillRanges();
		void addRangeMeas(int id1, int id2, float dist);
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
		std::vector <vertex_t> robots;
		std::vector<std::vector<float>> edges;
};