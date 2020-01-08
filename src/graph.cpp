#include "graph.h"

using namespace boost;
using namespace std;

// The clique_printer is a visitor that will print the vertices that comprise
// a clique. Note that the vertices are not given in any specific order.
template <typename OutputStream>
struct clique_printer
{
	clique_printer(OutputStream& stream)
	: os(stream)
	{ }

    template <typename Clique, typename Graph>
	void clique(const Clique& c, const Graph& g)
	{
        // Iterate over the clique and print each vertex within it.
		typename Clique::const_iterator i, end = c.end();
		for(i = c.begin(); i != end; ++i) {
			os << g[*i].name << " ";
		}
		os << endl;
	}
	OutputStream& os;
};

bool DistanceGraph::addEdge(vertex_t r1, vertex_t r2, float dist){
	edge_t e; bool b;
	boost::tie(e,b) = boost::add_edge(r1,r2,dist,g);
	return b;
}

bool DistanceGraph::removeEdge(vertex_t r1, vertex_t r2){
	boost::remove_edge(r1,r2,g);
}

vertex_t DistanceGraph::addVertex(Point2d startLoc){
	int id = num_vertices(g);
	vertex_t v = add_vertex(Robot(startLoc, id), g);
	return v;
}


void DistanceGraph::moveRobot(int id, float x, float y){
	g[id].move(x,y);
}

