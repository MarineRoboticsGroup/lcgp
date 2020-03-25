#include "graph.h"

using namespace scip;

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

bool DistanceGraph::addEdge(vertex_t r1, vertex_t r2, float dist){
	edge_t e; bool b;
	boost::remove_edge(r1, r2, g);
	boost::remove_edge(r2, r1, g);
	boost::tie(e,b) = boost::add_edge(r1,r2,dist,g);
	return b;
}

bool DistanceGraph::removeEdge(vertex_t r1, vertex_t r2){
	boost::remove_edge(r1,r2,g);
}

float DistanceGraph::getEdge(vertex_t r1, vertex_t r2){
	edge_t e; bool b;
	boost::tie(e,b) = boost::edge(r1,r2,g);
	if (b){	return g[e]; }
	boost::tie(e,b) = boost::edge(r2,r1,g);
	if (b) { return g[e]; }
	return -1;
}



vertex_t DistanceGraph::addVertex(Point2d startLoc, bool isRob){
	int id = boost::num_vertices(g);
	vertex_t v = boost::add_vertex(Robot(startLoc, id, isRob), g);
	return v;
}

vertex_t DistanceGraph::addVertex(Point2d startLoc, bool isRob, float stdDev){
	int id = boost::num_vertices(g);
	vertex_t v = boost::add_vertex(Robot(startLoc, id, isRob, stdDev), g);
	return v;
}


void DistanceGraph::moveRobot(int id, float x, float y){
	g[id].move(x,y);
}

SCIP_RETCODE DistanceGraph::realizeGraphSCIP(std::vector<Point2d> &locs){
	// Initialize Problem
	SCIP* scip;
	SCIP_VAR** y;
	SCIP_VAR** x;
	
	SCIP_CALL( SCIPcreate(& scip) );
	SCIP_CALL( SCIPincludeDefaultPlugins(scip) );
	SCIP_CALL( SCIPsetRealParam(scip, "limits/gap", 0.5) );
	SCIP_CALL( SCIPsetRealParam(scip, "limits/absgap", 0.05) );
	// SCIP_CALL( SCIPsetRealParam(scip, "limits/time", .35) );

	SCIP_CALL( setupProblem(scip, &x, &y, locs) );

	SCIP_CALL( SCIPsolve(scip) );

	if (0 < SCIPgetNSols(scip))
	{
		auto sol = SCIPgetBestSol(scip);
		for (int j = 0; j < boost::num_vertices(g); ++j)
		{
			auto xEst = SCIPgetSolVal(scip, sol, x[j]), yEst = SCIPgetSolVal(scip, sol, y[j]);
			locs[j] = Point2d(xEst, yEst);
			SCIP_CALL( SCIPreleaseVar(scip, &y[j]) );
			SCIP_CALL( SCIPreleaseVar(scip, &x[j]) );
		}
	}

	SCIPfreeMemoryArray(scip, &x);
	SCIPfreeMemoryArray(scip, &y);

	SCIP_CALL( SCIPfree(&scip) );
}

SCIP_RETCODE DistanceGraph::setupProblem(
    SCIP*                 scip,               /**< SCIP data structure */
    SCIP_VAR***           xvars,              /**< buffer to store pointer to x variables array */
    SCIP_VAR***           yvars,              /**< buffer to store pointer to y variables array */
	std::vector<Point2d> &estLocs
	) {

	SCIP_CALL(SCIPcreateProbBasic(scip, "graph_realization"));

	int nRobots = boost::num_vertices(g);
	int nMeas = boost::num_edges(g);

    // Names for variables
	char namex[SCIP_MAXSTRLEN];
	char namey[SCIP_MAXSTRLEN];
	char namer[SCIP_MAXSTRLEN];

	// Initial position constraints
	SCIP_Real x0 = g[0].getCurrLoc().getX();
	SCIP_Real y0 = g[0].getCurrLoc().getY();
	SCIP_Real x1 = g[1].getCurrLoc().getX();
	
	SCIP_Real xlb, xub, ylb, yub, range; 

	// Coefficients
	SCIP_Real minusone = -1.0;
	SCIP_Real plusone = 1.0;

	// variables:
	// * r[i] i=0..M, such that: value function=sum r[i]
	// * y[i] i=0..N, Y position of node 'i'
	// * x[i] i=0..N, Y position of node 'i'
	SCIP_VAR** x;
	SCIP_VAR** y;
	SCIP_VAR** r;

	// constraint
	SCIP_CONS* cons;
	char consname[SCIP_MAXSTRLEN];

	// Memory for positions
	SCIP_CALL( SCIPallocMemoryArray(scip, &x, (size_t) nRobots) );
	SCIP_CALL( SCIPallocMemoryArray(scip, &y, (size_t) nRobots) );
	*xvars = x;
	*yvars = y;

	// Memory for residual variable
	SCIP_CALL( SCIPallocBufferArray(scip, &r, (size_t) nMeas ) );

	// Construct and add x and y variables 
	for (int i = 0; i < nRobots; ++i)
	{
		// 5 sigma bounds on position std dev from last measurement
		range = 3 * std::max(g[i].getControlSat(), g[i].getStdDev());

		xlb = estLocs[i].getX() - range;
		xub = estLocs[i].getX() + range;
		ylb = estLocs[i].getY() - range;
		yub = estLocs[i].getY() + range;


		SCIPsnprintf(namex, SCIP_MAXSTRLEN, "x(%d)", i);
		SCIPsnprintf(namey, SCIP_MAXSTRLEN, "y(%d)", i);

		SCIP_CALL( SCIPcreateVarBasic(scip, &x[i], namex, xlb, xub, 0.0, SCIP_VARTYPE_CONTINUOUS) );
		SCIP_CALL( SCIPcreateVarBasic(scip, &y[i], namey, ylb, yub, 0.0, SCIP_VARTYPE_CONTINUOUS) );
		SCIP_CALL(SCIPaddVar(scip, x[i]));
		SCIP_CALL(SCIPaddVar(scip, y[i]));

		// Add x and y constraints
		if (i == 0)
		{
			SCIPsnprintf(consname, SCIP_MAXSTRLEN, "x%d_cons", i);
			SCIP_CALL( SCIPcreateConsBasicLinear(scip, &cons, consname, 1, &x[i], &plusone, x0, x0));
			SCIP_CALL( SCIPaddCons(scip, cons) );
			SCIP_CALL( SCIPreleaseCons(scip, &cons) );
			SCIPsnprintf(consname, SCIP_MAXSTRLEN, "y%d_cons", i);
			SCIP_CALL( SCIPcreateConsBasicLinear(scip, &cons, consname, 1, &y[i], &plusone, y0, y0));
			SCIP_CALL( SCIPaddCons(scip, cons) );
			SCIP_CALL( SCIPreleaseCons(scip, &cons) );
		}
	}

	for (int i = 0; i < nMeas; ++i)
	{
		SCIPsnprintf(namer, SCIP_MAXSTRLEN, "r(%d)", i);
		SCIP_CALL( SCIPcreateVarBasic(scip, &r[i], namer, 0, 999, 1, SCIP_VARTYPE_CONTINUOUS) );
		SCIP_CALL(SCIPaddVar(scip, r[i]));
	}

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

	// The actual distance measurement
	SCIP_Real d_meas = -1.0;

	int i = 0, a = 0, b = 0;
	auto es = boost::edges(g);

	for (auto eit = es.first; eit != es.second; ++eit)
	{

		// Get range measurement details
		d_meas = g[*eit];
		a = boost::source(*eit, g);
		b = boost::target(*eit, g);

		SCIP_VAR* varstoadd[4] = { x[a], y[a], x[b], y[b] };

		SCIP_MESSAGEHDLR *messagehdlr;

		// initialize child expressions with indices to assign variables later
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &x_a, SCIP_EXPR_VARIDX, 0) );
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &y_a, SCIP_EXPR_VARIDX, 1) );
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &x_b, SCIP_EXPR_VARIDX, 2) );
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &y_b, SCIP_EXPR_VARIDX, 3) );

		// create intermediate constraints
        // * expr1: (x_a - x_b)
        // * expr2: (x_a - x_b)^2
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &expr1, SCIP_EXPR_MINUS, x_a, x_b) );
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &expr2, SCIP_EXPR_SQUARE, expr1) );

        // * expr3: (y_a - y_b)
        // * expr4: (y_a - y_b)^2
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &expr3, SCIP_EXPR_MINUS, y_a, y_b) );
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &expr4, SCIP_EXPR_SQUARE, expr3) );

        // * expr5: (x_a - x_b)^2 + (y_a - y_b)^2
        // * expr6: sqrt( (x_a - x_b)^2 + (y_a - y_b)^2 )
        // * expr5: d - sqrt( (x_a - x_b)^2 + (y_a - y_b)^2 )
        // * expr6: ( d - sqrt( (x_a - x_b)^2 + (y_a - y_b)^2 ) )^2
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &expr5, SCIP_EXPR_PLUS, expr4, expr2) );
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &expr6, SCIP_EXPR_SQRT, expr5) );
		SCIP_CALL( SCIPexprCreateLinear(SCIPblkmem(scip), &expr7, 1, &expr6, &minusone, d_meas) );
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &expr8, SCIP_EXPR_SQUARE, expr7) );

		// Create Expression Tree
		SCIP_CALL( SCIPexprtreeCreate(SCIPblkmem(scip), &exprtree, expr8, 4, 0, NULL) );
		SCIP_CALL( SCIPexprtreeSetVars(exprtree, 4, varstoadd) );

		// Create Constraint
		SCIPsnprintf(consname, SCIP_MAXSTRLEN, "constraint(%d)", i);
		SCIP_CALL( SCIPcreateConsBasicNonlinear(scip, &cons, consname, 1, &r[i], &plusone, 1, &exprtree, &minusone, 0.0, SCIPinfinity(scip)) );

		// Add constraint and free constraint and tree
		SCIP_CALL( SCIPaddCons(scip, cons) );
		SCIP_CALL( SCIPreleaseCons(scip, &cons) );
		SCIP_CALL( SCIPexprtreeFree(&exprtree) );
		SCIP_CALL( SCIPreleaseVar(scip, &r[i]) );
		i++;
	}

	SCIPfreeBufferArray(scip, &r);

	return SCIP_OKAY;
}
