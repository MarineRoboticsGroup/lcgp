#ifndef GRAPH_H_
#define GRAPH_H_ 


#include "helper.hpp"
#include "robot.h"

#include <iostream>
#include <boost/config.hpp>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/bron_kerbosch_all_cliques.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>

typedef boost::adjacency_list < boost::listS, boost::vecS, boost::undirectedS,
	Robot, float > graph_t;
// typedef boost::adjacency_list < boost::listS, boost::vecS, boost::undirectedS,
// 	boost::no_property, boost::property < boost::edge_weight_t, float> > graph_t;
typedef boost::graph_traits < graph_t >::vertex_descriptor vertex_descriptor;
typedef boost::graph_traits < graph_t >::edge_descriptor edge_descriptor;
typedef std::pair<int, int> Edge;


class DistanceGraph
{
public:
	
	void runSample();
	bool addEdge(int id1, int id2, float dist);
	Robot& addVertex(Robot r);


private:
	graph_t graph();
};

#endif
