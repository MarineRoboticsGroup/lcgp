#include "graph.h"

using namespace scip;

bool DistanceGraph::addEdge(vertex_t r1, vertex_t r2, float dist){
	edge_t e; bool b;
	boost::tie(e,b) = boost::add_edge(r1,r2,dist,g);
	return b;
}

bool DistanceGraph::removeEdge(vertex_t r1, vertex_t r2){
	boost::remove_edge(r1,r2,g);
}

vertex_t DistanceGraph::addVertex(Point2d startLoc, bool isRob){
	int id = boost::num_vertices(g);
	vertex_t v = boost::add_vertex(Robot(startLoc, id, isRob), g);
	return v;
}


void DistanceGraph::moveRobot(int id, float x, float y){
	g[id].move(x,y);
}

SCIP_RETCODE realizeGraph(){
	int nRobots = 10;

	SCIP* sc = NULL;
	SCIP_CALL(SCIPcreate(& sc));
	SCIP_CALL(SCIPincludeDefaultPlugins(sc));


	SCIP_CALL(SCIPcreateProb(sc, "graph_realization", NULL, NULL,
		NULL, NULL, NULL, NULL, NULL));

	std::vector<SCIP_VAR*> xPos(nRobots), yPos(nRobots);

	for (int i = 0; i < nRobots; ++i)
	{
		std::string x = "x" + std::to_string(i), y = "y" + std::to_string(i);

		// Add x values
		SCIP_CALL(SCIPcreateVar(sc, & xPos[i], x.c_str(), 0.0, 1.0, 1.0, 
           SCIP_VARTYPE_CONTINUOUS, TRUE, FALSE,
            NULL, NULL, NULL, NULL, NULL));
		SCIP_CALL(SCIPaddVar(sc, xPos[i]));

		// Add y values
		SCIP_CALL(SCIPcreateVar(sc, & yPos[i], y.c_str(), 0.0, 1.0, 1.0,
            SCIP_VARTYPE_CONTINUOUS, TRUE, FALSE,
            NULL, NULL, NULL, NULL, NULL));
		SCIP_CALL(SCIPaddVar(sc, yPos[i]));
	}

	

}


// static
//  SCIP_RETCODE runSCIP(
//     int                        argc,          /**< number of arguments from the shell */
//     char**                     argv           /**< array of shell arguments */
//     )
//  {
//     SCIP* sc = NULL;
 
 
//     /*********
//      * Setup *
//      *********/
 
//     /* initialize SCIP */
//     SCIP_CALL( SCIPcreate(&sc) );
 
//     /* we explicitly enable the use of a debug solution for this main SCIP instance */
//     SCIPenableDebugSol(sc);
 
//     /* include TSP specific plugins */
//     SCIP_CALL( SCIPincludeObjReader(sc, new ReaderTSP(sc), TRUE) );
//     SCIP_CALL( SCIPincludeObjConshdlr(sc, new ConshdlrSubtour(sc), TRUE) ); 
//     SCIP_CALL( SCIPincludeObjEventhdlr(sc, new EventhdlrNewSol(sc), TRUE) );
//     SCIP_CALL( SCIPincludeObjHeur(sc, new HeurFarthestInsert(sc), TRUE) );
//     SCIP_CALL( SCIPincludeObjHeur(sc, new Heur2opt(sc), TRUE) );
//     SCIP_CALL( SCIPincludeObjHeur(sc, new HeurFrats(sc), TRUE) );
 
//     /* include default SCIP plugins */
//     SCIP_CALL( SCIPincludeDefaultPlugins(sc) );
 
 
//     /**********************************
//      * Process command line arguments *
//      **********************************/
 
//     SCIP_CALL( SCIPprocessShellArguments(sc, argc, argv, "sctsp.set") );
 
 
//     /********************
//      * Deinitialization *
//      ********************/
 
//     SCIP_CALL( SCIPfree(&sc) );
 
//     BMScheckEmptyMemory();
 
//     return SCIP_OKAY;
//  }
 
//  /** main method starting TSP code */
//  int main(
//     int                        argc,          /**< number of arguments from the shell */
//     char**                     argv           /**< array of shell arguments */
//     )
//  {
//     SCIP_RETCODE retcode;
 
//     retcode = runSCIP(argc, argv);
//     if( retcode != SCIP_OKAY )
//     {
//        SCIPprintError(retcode);
//        return -1;
//     }
 
//     return 0;
//  }


