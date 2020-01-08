#ifndef GRAPH_H_
#define GRAPH_H_ 


#include "helper.hpp"
#include "robot.h"

#include <iostream>
#include <boost/config.hpp>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/bron_kerbosch_all_cliques.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>

typedef boost::adjacency_list < boost::listS, boost::vecS, boost::undirectedS,
	Robot, float > Graph;
typedef boost::graph_traits < Graph >::vertex_descriptor vertex_t;
typedef boost::graph_traits < Graph >::edge_descriptor edge_t;
typedef std::pair<Robot, Robot> Edge;


class DistanceGraph
{
private:
	Graph g;

public:

	// Graph level controls
	bool addEdge(vertex_t r1, vertex_t r2, float dist);
	bool removeEdge(vertex_t r1, vertex_t r2);
	vertex_t addVertex(Point2d startLoc);
	Robot& getVertex(int id) {return g[id];}

	// Robot level controls
	void moveRobot(int id, float x, float y);


};

#endif
