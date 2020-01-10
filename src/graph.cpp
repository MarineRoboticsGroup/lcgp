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

void DistanceGraph::printInfo(){
	std::cout << "iterate over vertices, then over its neighbors\n";
	auto vs = boost::vertices(g);
	for (auto vit = vs.first; vit != vs.second; ++vit) {
		auto neighbors = boost::adjacent_vertices(*vit, g);
		for (auto nit = neighbors.first; nit != neighbors.second; ++nit)
			std::cout << *vit << ' ' << *nit << std::endl;
	}

	std::cout << "iterate directly over edges\n";
	auto es = boost::edges(g);
	for (auto eit = es.first; eit != es.second; ++eit) {
		std::cout << boost::source(*eit, g) << " -> "  << boost::target(*eit, g) << ": " << g[*eit] << std::endl;
	}
}


vertex_t DistanceGraph::addVertex(Point2d startLoc, bool isRob){
	int id = boost::num_vertices(g);
	vertex_t v = boost::add_vertex(Robot(startLoc, id, isRob), g);
	return v;
}


void DistanceGraph::moveRobot(int id, float x, float y){
	g[id].move(x,y);
}

SCIP_RETCODE DistanceGraph::realizeGraphSCIP(){
	std::cout << "simplify expression: ";
    SCIPdebugMessage("simplify expression: ");
	int nRobots = boost::num_vertices(g);
	int nMeas = boost::num_edges(g);

	// Initialize Problem
	SCIP* scip;
	SCIP_CALL(SCIPcreate(& scip));
	SCIP_CALL(SCIPincludeDefaultPlugins(scip));
	SCIP_CALL(SCIPcreateProb(scip, "graph_realization", NULL, NULL,
		NULL, NULL, NULL, NULL, NULL));

	// set gap at which SCIP will stop 
	SCIP_CALL( SCIPsetRealParam(scip, "limits/gap", 0.05) );

    // Names for variables
	char namex[SCIP_MAXSTRLEN];
	char namey[SCIP_MAXSTRLEN];
	char namer[SCIP_MAXSTRLEN];

	// Initial position constraints
	SCIP_Real x0 = 0;
	SCIP_Real y0 = 0;
	SCIP_Real x1 = x0;


     // variables:
     // * r[i] i=0..M, such that: value function=sum r[i]
     // * y[i] i=0..N, Y position of node 'i'
     // * x[i] i=0..N, Y position of node 'i'
	SCIP_VAR** x;
	SCIP_VAR** y;
	SCIP_VAR** r;

	SCIP_CALL( SCIPallocMemoryArray(scip, &x, (size_t) nRobots) );
	SCIP_CALL( SCIPallocMemoryArray(scip, &y, (size_t) nRobots) );

	// Construct and add x and y variables 
	for (int i = 0; i < nRobots; ++i)
	{
		SCIPsnprintf(namex, SCIP_MAXSTRLEN, "x(%d)", i);
		SCIPsnprintf(namey, SCIP_MAXSTRLEN, "y(%d)", i);

		// Add x and y values
		if (i == 0)
		{
			SCIP_CALL( SCIPcreateVarBasic(scip, &x[i], namex, x0, x0, 0.0, SCIP_VARTYPE_CONTINUOUS) );
			SCIP_CALL( SCIPcreateVarBasic(scip, &y[i], namey, y0, y0, 0.0, SCIP_VARTYPE_CONTINUOUS) );
		}
		else if (i == 1)
		{
			SCIP_CALL( SCIPcreateVarBasic(scip, &x[i], namex, x1, x1, 0.0, SCIP_VARTYPE_CONTINUOUS) );
			SCIP_CALL( SCIPcreateVarBasic(scip, &y[i], namey, -SCIPinfinity(scip), SCIPinfinity(scip), 0.0, SCIP_VARTYPE_CONTINUOUS) );
		}
		else {
			SCIP_CALL( SCIPcreateVarBasic(scip, &x[i], namex, -SCIPinfinity(scip), SCIPinfinity(scip), 0.0, SCIP_VARTYPE_CONTINUOUS) );
			SCIP_CALL( SCIPcreateVarBasic(scip, &y[i], namey, -SCIPinfinity(scip), SCIPinfinity(scip), 0.0, SCIP_VARTYPE_CONTINUOUS) );
		}

		SCIP_CALL(SCIPaddVar(scip, x[i]));
		SCIP_CALL(SCIPaddVar(scip, y[i]));
	}

	// construct constraint
	SCIP_CONS* cons;
	char consname[SCIP_MAXSTRLEN];

	// All the expressions necessary for constraint
	SCIP_EXPR* x_a;
	SCIP_EXPR* y_a;
	SCIP_EXPR* x_b;
	SCIP_EXPR* y_b;
	SCIP_EXPR* expr1;
	SCIP_EXPR* expr2;
	SCIP_EXPR* expr3;
	SCIP_EXPR* expr4;
	SCIP_EXPR* expr5;
	SCIP_EXPR* expr6;
	SCIP_EXPR* expr7;
	SCIP_EXPR* expr8;

	// Expression tree for nonlinear constraint
	SCIP_EXPRTREE* exprtree;

	// Coefficients
	SCIP_Real minusone = -1.0;
	SCIP_Real plusone = 1.0;

	// The actual distance measurement
	SCIP_Real d_meas = -1.0;

	// Memory for residual variable
	SCIP_CALL( SCIPallocBufferArray(scip, &r, (size_t) nMeas  ) );

	int i = 0, a = 0, b = 0;
	auto es = boost::edges(g);
	for (auto eit = es.first; eit != es.second; ++eit)
	{
		SCIPsnprintf(namer, SCIP_MAXSTRLEN, "r(%d)", i);
		SCIP_CALL( SCIPcreateVarBasic(scip, &r[i], namer, 0.0, SCIPinfinity(scip), 1, SCIP_VARTYPE_CONTINUOUS) );

		// Get range measurement details
		d_meas = g[*eit];
		a = boost::source(*eit, g);
		b = boost::target(*eit, g);

		SCIP_VAR* varstoadd[4] = { x[a], y[a], x[b], y[b] };

		// initialize child expressions with indices to assign variables later
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &x_a, SCIP_EXPR_VARIDX, 0) );
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &y_a, SCIP_EXPR_VARIDX, 1) );
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &x_b, SCIP_EXPR_VARIDX, 2) );
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &y_b, SCIP_EXPR_VARIDX, 3) );

		// create intermediate constraints
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &expr1, SCIP_EXPR_MINUS, x_a, x_b) );
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &expr2, SCIP_EXPR_SQUARE, expr1) );
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &expr3, SCIP_EXPR_MINUS, y_a, y_b) );
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &expr4, SCIP_EXPR_SQUARE, expr3) );
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &expr5, SCIP_EXPR_PLUS, expr4, expr2) );
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &expr6, SCIP_EXPR_SQRT, expr5) );
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &expr7, SCIP_EXPR_MINUS, d_meas, expr6) );
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &expr8, SCIP_EXPR_SQUARE, expr7) );

		// Create Expression Tree
		SCIP_CALL( SCIPexprtreeCreate(SCIPblkmem(scip), &exprtree, expr8, 4, 0, NULL) );
		SCIP_CALL( SCIPexprtreeSetVars(exprtree, 4, varstoadd) );


		FILE *fp = fopen("./test.txt", "w+");		
		SCIP_MESSAGEHDLR *messagehdlr;
		SCIPdebug( SCIPexprPrint(expr1, messagehdlr, fp, NULL, NULL, NULL) );	
		SCIP_CALL( SCIPexprtreePrintWithNames(exprtree, messagehdlr, fp) );	

		// Create Constraint
		SCIPsnprintf(consname, SCIP_MAXSTRLEN, "constraint(%d)", i);
		SCIP_CALL( SCIPcreateConsBasicNonlinear(scip, &cons, consname, 1, &r[i], &plusone, 1, &exprtree, &minusone, 0.0, SCIPinfinity(scip)) );

		// Add constraint and free constraint and tree
		SCIP_CALL( SCIPaddCons(scip, cons) );
		SCIP_CALL( SCIPreleaseCons(scip, &cons) );
		SCIP_CALL( SCIPexprtreeFree(&exprtree) );\
		i++;
	}

     // release intermediate variables 
	for( i = 0; i < nRobots; ++i )
	{
		SCIP_CALL( SCIPreleaseVar(scip, &r[i]) );
	}

	SCIPfreeBufferArray(scip, &r);

	SCIPinfoMessage(scip, NULL, "Original problem:\n");
	SCIP_CALL( SCIPprintOrigProblem(scip, NULL, "cip", FALSE) );

	SCIPinfoMessage(scip, NULL, "\nSolving...\n");
	SCIP_CALL( SCIPsolve(scip) );

	if( SCIPgetNSols(scip) > 0 )
	{
		SCIPinfoMessage(scip, NULL, "\nSolution:\n");
		SCIP_CALL( SCIPprintSol(scip, SCIPgetBestSol(scip), NULL, FALSE) );
	}
}






