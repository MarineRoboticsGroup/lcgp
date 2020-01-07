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


void DistanceGraph::runSample(){
/*	
	// Declare the graph type and its vertex and edge types.
	typedef undirected_graph<Actor> Graph;
	typedef graph_traits<Graph>::vertex_descriptor Vertex;
	typedef graph_traits<Graph>::edge_descriptor Edge;

	// The name map provides an abstract accessor for the names of
	// each vertex. This is used during graph creation.
	typedef property_map<Graph, string Actor::*>::type NameMap;

    // Create the graph and and its name map accessor.
	Graph g;
	NameMap nm(get(&Actor::name, g));

    // Read the graph from standard input.
	read_graph(g, nm, cin);

    // Instantiate the visitor for printing cliques
	clique_printer<ostream> vis(cout);

    // Use the Bron-Kerbosch algorithm to find all cliques, printing them
    // as they are found.
	bron_kerbosch_all_cliques(g, vis);
*/

/*	const int num_nodes = 5;
	enum nodes { A, B, C, D, E };
	char name[] = "ABCDE";
	Edge edge_array[] = { Edge(A, C), Edge(B, B), Edge(B, D), Edge(B, E),
		Edge(C, B), Edge(C, D), Edge(D, E), Edge(E, A), Edge(E, B)
	};
	float weights[] = { 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5 };
	int num_arcs = sizeof(edge_array) / sizeof(Edge);
	
	graph_t g(edge_array, edge_array + num_arcs, weights, num_nodes);
	property_map<graph_t, edge_weight_t>::type weightmap = get(edge_weight, g);
	std::vector<vertex_descriptor> p(num_vertices(g));
	std::vector<int> d(num_vertices(g));
	vertex_descriptor s = vertex(A, g);

	
	dijkstra_shortest_paths(g, s, predecessor_map(&p[0]).distance_map(&d[0]));
	std::cout << "distances and parents:" << std::endl;
	graph_traits < graph_t >::vertex_iterator vi, vend;
	for (tie(vi, vend) = vertices(g); vi != vend; ++vi) {
		std::cout << "distance(" << name[*vi] << ") = " << d[*vi] << ", ";
		std::cout << "parent(" << name[*vi] << ") = " << name[p[*vi]] << std::
		endl;
	}
	std::cout << std::endl;*/


}

bool DistanceGraph::addEdge(int id1, int id2, float dist){
	boost::add_edge(id1, id2, dist, graph);
}
