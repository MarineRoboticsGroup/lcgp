#include "graph.h"

using namespace scip;


namespace {
	static
	SCIP_RETCODE setupProblem(
    SCIP*                 scip,               /**< SCIP data structure */
    unsigned int          n,                  /**< number of points for discretization */
    SCIP_Real*            coord,              /**< array containing [y(0), y(N), x(0), x(N)] */
    SCIP_VAR***           xvars,              /**< buffer to store pointer to x variables array */
    SCIP_VAR***           yvars               /**< buffer to store pointer to y variables array */
		)
	{


	}
}

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
	SCIP_CALL(SCIPcreateProbBasic(scip, "graph_realization"));

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
	SCIP_Real diff[2] = {-1.0, 1.0};


	// The actual distance measurement
	SCIP_Real d_meas = -1.0;

	// Memory for residual variable
	SCIP_CALL( SCIPallocBufferArray(scip, &r, (size_t) nMeas  ) );

	int i = 0, a = 0, b = 0;
	auto es = boost::edges(g);
	FILE *fp = fopen("./test.txt", "w");		

	for (auto eit = es.first; eit != es.second; ++eit)
	{
		SCIPsnprintf(namer, SCIP_MAXSTRLEN, "r(%d)", i);
		SCIP_CALL( SCIPcreateVarBasic(scip, &r[i], namer, 0.0, SCIPinfinity(scip), 1, SCIP_VARTYPE_CONTINUOUS) );

		// Get range measurement details
		d_meas = g[*eit];
		// d_meas = 4.0;
		a = boost::source(*eit, g);
		b = boost::target(*eit, g);

		SCIP_VAR* varstoadd[4] = { x[a], y[a], x[b], y[b] };

		SCIP_MESSAGEHDLR *messagehdlr;

		// initialize child expressions with indices to assign variables later
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &x_a, SCIP_EXPR_VARIDX, 0) );
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &y_a, SCIP_EXPR_VARIDX, 1) );
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &x_b, SCIP_EXPR_VARIDX, 2) );
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &y_b, SCIP_EXPR_VARIDX, 3) );
		SCIP_EXPR* xs[2] = { x_a, x_b};
		SCIP_EXPR* ys[2] = { x_a, x_b};


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
		// SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &expr7, SCIP_EXPR_MINUS, d_meas, expr6) );
		SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &expr8, SCIP_EXPR_SQUARE, expr7) );

		// SCIPprintMemoryDiagnostic(scip);

		// Create Expression Tree
		SCIP_CALL( SCIPexprtreeCreate(SCIPblkmem(scip), &exprtree, expr8, 4, 0, NULL) );
		SCIP_CALL( SCIPexprtreeSetVars(exprtree, 4, varstoadd) );
		if (i == 0)
		{
			SCIP_CALL( SCIPexprtreePrintWithNames(exprtree, SCIPgetMessagehdlr(scip), fp) );
		}

		// Create Constraint
		SCIPsnprintf(consname, SCIP_MAXSTRLEN, "constraint(%d)", i);
		SCIP_CALL( SCIPcreateConsBasicNonlinear(scip, &cons, consname, 1, &r[i], &plusone, 1, &exprtree, &minusone, 0.0, SCIPinfinity(scip)) );

		// Add constraint and free constraint and tree
		SCIP_CALL( SCIPaddCons(scip, cons) );
		i++;
		// SCIP_CALL( SCIPreleaseCons(scip, &cons) );
		// SCIP_CALL( SCIPexprtreeFree(&exprtree) );
	}
	fclose (fp);

     // release intermediate variables 
	// for( i = 0; i < nRobots; ++i )
	// {
	// 	SCIP_CALL( SCIPreleaseVar(scip, &r[i]) );
	// }

	// SCIPfreeBufferArray(scip, &r);

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

int DistanceGraph::realizeGraphIPOPT(){
	   // Create an instance of your nlp...
	SmartPtr<TNLP> mynlp = new MyNLP();

   // Create an instance of the IpoptApplication
   //
   // We are using the factory, since this allows us to compile this
   // example with an Ipopt Windows DLL
	SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
	app->Options()->SetStringValue("mu_strategy", "adaptive");
	app->Options()->SetStringValue("output_file", "ipopt.out");
	app->Options()->SetStringValue("linear_solver", "mumps");

   // Initialize the IpoptApplication and process the options
	ApplicationReturnStatus status;
	status = app->Initialize();
	if( status != Solve_Succeeded )
	{
		std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
		return (int) status;
	}

	status = app->OptimizeTNLP(mynlp);

	if( status == Solve_Succeeded )
	{
      // Retrieve some statistics about the solve
		Index iter_count = app->Statistics()->IterationCount();
		std::cout << std::endl << std::endl << "*** The problem solved in " << iter_count << " iterations!" << std::endl;

		Number final_obj = app->Statistics()->FinalObjective();
		std::cout << std::endl << std::endl << "*** The final value of the objective function is " << final_obj << '.'
		<< std::endl;
	}

	return (int) status;

}





