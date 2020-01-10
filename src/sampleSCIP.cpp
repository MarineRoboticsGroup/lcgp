 /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
 /*                                                                           */
 /*                  This file is part of the program and library             */
 /*         SCIP --- Solving Constraint Integer Programs                      */
 /*                                                                           */
 /*    Copyright (C) 2002-2019 Konrad-Zuse-Zentrum                            */
 /*                            fuer Informationstechnik Berlin                */
 /*                                                                           */
 /*  SCIP is distributed under the terms of the ZIB Academic License.         */
 /*                                                                           */
 /*  You should have received a copy of the ZIB Academic License              */
 /*  along with SCIP; see the file COPYING. If not visit scip.zib.de.         */
 /*                                                                           */
 /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
 
 /**@file   brachistochrone.c
  * @brief  Computing a minimum-time trajectory for a particle to move from point A to B under gravity only
  * @author Anass Meskini
  * @author Stefan Vigerske
  *
  * This is an example that uses expressions and expression trees to set up non-linear constraints in SCIP when used as
  * a callable library. This example implements a discretized model to obtain the trajectory associated with the shortest
  * time to go from point A to B for a particle under gravity only.
  *
  * The model:
  *
  * Given \f$N\f$ number of points for the discretisation of the trajectory, we can approximate the time to go from
  * \f$(x_0,y_0)\f$ to \f$ (x_N,y_N)\f$ for a given trajectory by \f[T = \sqrt{\frac{2}{g}}
  * \sum_{0}^{N-1} \frac{\sqrt{(y_{i+1} - y_i)^2 + (x_{i+1} - x_i)^2}}{\sqrt{1-y_{i+1}} + \sqrt{1 - y_i}}.\f]
  * We seek to minimize \f$T\f$.
  * A more detailed description of the model can be found in the brachistochrone directory of http://scip.zib.de/workshop2018/pyscipopt-exercises.tgz
  *
  * Passing this equation as it is to SCIP does not lead to satisfying results, though, so we reformulate a bit.
  * Let \f$t_i \geq \frac{\sqrt{(y_{i+1} - y_i)^2 + (x_{i+1} - x_i)^2}}{\sqrt{1-y_{i+1}} + \sqrt{1 - y_i}}\f$.
  * To avoid a potential division by zero, we rewrite this as
  * \f$t_i (\sqrt{1-y_{i+1}} + \sqrt{1 - y_i}) \geq \sqrt{(y_{i+1} - y_i)^2 + (x_{i+1} - x_i)^2}, t_i\geq 0\f$.
  * Further, introduce \f$v_i \geq 0\f$ such that \f$v_i \geq \sqrt{(y_{i+1} - y_i)^2 + (x_{i+1} - x_i)^2}\f$.
  * Then we can state the optimization problem as
  * \f{align}{ \min \;& \sqrt{\frac{2}{g}} \sum_{i=0}^{N-1} t_i \\
  *   \text{s.t.} \; & t_i (\sqrt{1-y_{i+1}} + \sqrt{1 - y_i}) \geq v_i \\
  *     & v_i^2 \geq (y_{i+1} - y_i)^2 + (x_{i+1} - x_i)^2 \\
  *     & t_i \geq 0,\; v_i \geq 0 \\
  *     & i = 0, \ldots, N-1
  * \f}
  *
  * Further, we can require that the particle moves only in direction horizontally, that is
  * \f$x_i \leq x_{i+1}\f$ if \f$x_0 \leq x_N\f$, and \f$x_{i+1} \leq x_{i}\f$, otherwise,
  * and that it will not move higher than the start-coordinate.
  */
 
 /*--+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
 
 #include <stdio.h>
 #include <stdlib.h>
 
 #include "scip/pub_misc.h"
 #include "scip/scip.h"
 #include "scip/scipdefplugins.h"
 
 /* default start and end points */
 #define Y_START  1.0
 #define Y_END    0.0
 #define X_START  0.0
 #define X_END   10.0
 
 /** sets up problem */
 static
 SCIP_RETCODE setupProblem(
    SCIP*                 scip,               /**< SCIP data structure */
    unsigned int          n,                  /**< number of points for discretization */
    SCIP_Real*            coord,              /**< array containing [y(0), y(N), x(0), x(N)] */
    SCIP_VAR***           xvars,              /**< buffer to store pointer to x variables array */
    SCIP_VAR***           yvars               /**< buffer to store pointer to y variables array */
    )
 {
    /* variables:
     * t[i] i=0..N-1, such that: value function=sum t[i]
     * v[i] i=0..N-1, such that: v_i = ||(x_{i+1},y_{i+1})-(x_i,y_i)||_2
     * y[i] i=0..N, projection of trajectory on y-axis
     * x[i] i=0..N, projection of trajectory on x-axis
     */
    SCIP_VAR** t;
    SCIP_VAR** v;
    SCIP_VAR** x;
    SCIP_VAR** y;
 
    char namet[SCIP_MAXSTRLEN];
    char namev[SCIP_MAXSTRLEN];
    char namey[SCIP_MAXSTRLEN];
    char namex[SCIP_MAXSTRLEN];
 
    SCIP_Real ylb;
    SCIP_Real yub;
    SCIP_Real xlb;
    SCIP_Real xub;
 
    /* an upper bound for v */
    SCIP_Real maxdistance = 10.0 * sqrt(SQR(coord[1]-coord[0]) + SQR(coord[3]-coord[2]));
    unsigned int i;
 
    /* create empty problem */
    SCIP_CALL( SCIPcreateProbBasic(scip, "brachistochrone") );
 
    /* allocate arrays of SCIP_VAR* */
    SCIP_CALL( SCIPallocBufferArray(scip, &t, (size_t) n ) );
    SCIP_CALL( SCIPallocBufferArray(scip, &v, (size_t) n ) );
    SCIP_CALL( SCIPallocMemoryArray(scip, &y, (size_t) n + 1) );
    SCIP_CALL( SCIPallocMemoryArray(scip, &x, (size_t) n + 1) );
    *xvars = x;
    *yvars = y;
 
    /* create and add variables to the problem and set the initial and final point constraints through upper and lower
     * bounds
     */
    for( i = 0; i < n+1; ++i )
    {
       /* setting up the names of the variables */
       if( i < n )
       {
          SCIPsnprintf(namet, SCIP_MAXSTRLEN, "t(%d)", i);
          SCIPsnprintf(namev, SCIP_MAXSTRLEN, "v(%d)", i);
       }
       SCIPsnprintf(namey, SCIP_MAXSTRLEN, "y(%d)", i);
       SCIPsnprintf(namex, SCIP_MAXSTRLEN, "x(%d)", i);
 
       /* fixing y(0), y(N), x(0), x(N) through lower and upper bounds */
       if( i == 0 )
       {
          ylb = coord[0];
          yub = coord[0];
          xlb = coord[2];
          xub = coord[2];
       }
       else if( i == n )
       {
          ylb = coord[1];
          yub = coord[1];
          xlb = coord[3];
          xub = coord[3];
       }
       else
       {
          /* constraint the other variables to speed up solving */
          ylb = -SCIPinfinity(scip);
          yub = coord[0];
          xlb = MIN(coord[2], coord[3]);
          xub = MAX(coord[2], coord[3]);
       }
 
       if( i < n )
       {
          SCIP_CALL( SCIPcreateVarBasic(scip, &t[i], namet, 0.0, SCIPinfinity(scip), sqrt(2.0/9.80665), SCIP_VARTYPE_CONTINUOUS) );
          SCIP_CALL( SCIPcreateVarBasic(scip, &v[i], namev, 0.0, maxdistance, 0.0, SCIP_VARTYPE_CONTINUOUS) );
       }
       SCIP_CALL( SCIPcreateVarBasic(scip, &y[i], namey, ylb, yub, 0.0, SCIP_VARTYPE_CONTINUOUS) );
       SCIP_CALL( SCIPcreateVarBasic(scip, &x[i], namex, xlb, xub, 0.0, SCIP_VARTYPE_CONTINUOUS) );
 
       if( i < n )
       {
          SCIP_CALL( SCIPaddVar(scip, t[i]) );
          SCIP_CALL( SCIPaddVar(scip, v[i]) );
       }
       SCIP_CALL( SCIPaddVar(scip, y[i]) );
       SCIP_CALL( SCIPaddVar(scip, x[i]) );
    }
 
    /* add constraints
     *
     * t(i) * sqrt(1 - y(i+1)) + t(i) * sqrt(1 - y(i)) >= v(i)
     * v(i)^2 >= (y(i+1) - y(i))^2 + (x(i+1) - x(i))^2
     * in the loop, create the i-th constraint
     */
    {
       SCIP_CONS* cons;
       char consname[SCIP_MAXSTRLEN];
 
       /* child expressions:
        * yplusexpr: expression for y[i+1]
        * yexpr: expression for y[i]
        * texpr, texpr2: expression for t[i]
        */
       SCIP_EXPR* yplusexpr;
       SCIP_EXPR* yexpr;
       SCIP_EXPR* texpr;
       SCIP_EXPR* texpr2;
 
       /* intermediary expressions */
       SCIP_EXPR* expr1;
       SCIP_EXPR* expr2;
       SCIP_EXPR* expr3;
       SCIP_EXPR* expr4;
       SCIP_EXPR* expr5;
       SCIP_EXPR* expr6;
 
       /* trees to hold the non-linear parts of the constraint */
       SCIP_EXPRTREE* exprtree[2];
 
       SCIP_Real minusone = -1.0;
 
       /* at each iteration create an expression for the non-linear part of the i-th constraint and add it the problem */
       for( i = 0; i < n; ++i )
       {
          /* vars to be added to the exprtree in this step of the loop */
          SCIP_VAR* varstoadd[3] = { y[i+1], y[i], t[i] };
 
          /* create expressions meant to be child expressions in the tree. give different indexes to the expressions to
           * assign the correct variables to them later
           */
          SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &yplusexpr, SCIP_EXPR_VARIDX, 0) );
          SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &yexpr, SCIP_EXPR_VARIDX, 1) );
          SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &texpr, SCIP_EXPR_VARIDX, 2) );
          SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &texpr2, SCIP_EXPR_VARIDX, 2) );
 
          /* set up the i-th constraint
           * expr1: 1 - y[i+1]
           * expr2: 1 - y[i]
           * expr3: sqrt(1 - y[i+1])
           * expr4: sqrt(1 - y[i])
           * expr5: t[i] * sqrt(1 - y[i+1])
           * expr6: t[i] * sqrt(1 - y[i])
           */
          SCIP_CALL( SCIPexprCreateLinear(SCIPblkmem(scip), &expr1, 1, &yplusexpr, &minusone, 1.0) );
          SCIP_CALL( SCIPexprCreateLinear(SCIPblkmem(scip), &expr2, 1, &yexpr, &minusone, 1.0) );
          SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &expr3, SCIP_EXPR_SQRT, expr1) );
          SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &expr4, SCIP_EXPR_SQRT, expr2) );
          SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &expr5, SCIP_EXPR_MUL, expr3, texpr) );
          SCIP_CALL( SCIPexprCreate(SCIPblkmem(scip), &expr6, SCIP_EXPR_MUL, expr4, texpr2) );
 
          /* create the expression trees with expr5 and expr6 as root */
          SCIP_CALL( SCIPexprtreeCreate(SCIPblkmem(scip), &exprtree[0], expr5, 3, 0, NULL) );
          SCIP_CALL( SCIPexprtreeCreate(SCIPblkmem(scip), &exprtree[1], expr6, 3, 0, NULL) );
          SCIP_CALL( SCIPexprtreeSetVars(exprtree[0], 3, varstoadd) );
          SCIP_CALL( SCIPexprtreeSetVars(exprtree[1], 3, varstoadd) );
 
          /* use the tree and a linear term to add the constraint exprtree0 + exprtree1 - v_i >= 0 */
          SCIPsnprintf(consname, SCIP_MAXSTRLEN, "timestep(%d)", i);
          SCIP_CALL( SCIPcreateConsBasicNonlinear(scip, &cons, consname, 1, &v[i],
                                                  &minusone, 2, exprtree, NULL, 0.0, SCIPinfinity(scip)) );
 
          /* add the constraint to the problem, release the constraint and free the trees */
          SCIP_CALL( SCIPaddCons(scip, cons) );
          SCIP_CALL( SCIPreleaseCons(scip, &cons) );
          SCIP_CALL( SCIPexprtreeFree(&exprtree[0]) );
          SCIP_CALL( SCIPexprtreeFree(&exprtree[1]) );
 
          /* add constraint v_i^2 >= (y_{i+1}^2 - 2*y_{i+1}y_i + y_i^2) + (x_{i+1}^2 - 2*x_{i+1}x_i + x_i^2)
           * SCIP should recognize that this can be formulated as SOC and do this reformulation
           */
          SCIPsnprintf(consname, SCIP_MAXSTRLEN, "steplength(%d)", i);
          SCIP_CALL( SCIPcreateConsBasicQuadratic(scip, &cons, consname, 0, NULL, NULL, 0, NULL, NULL, NULL, -SCIPinfinity(scip), 0.0) );
          SCIP_CALL( SCIPaddSquareCoefQuadratic(scip, cons, y[i], 1.0) );
          SCIP_CALL( SCIPaddSquareCoefQuadratic(scip, cons, y[i+1], 1.0) );
          SCIP_CALL( SCIPaddSquareCoefQuadratic(scip, cons, x[i], 1.0) );
          SCIP_CALL( SCIPaddSquareCoefQuadratic(scip, cons, x[i+1], 1.0) );
          SCIP_CALL( SCIPaddBilinTermQuadratic(scip, cons, y[i], y[i+1], -2.0) );
          SCIP_CALL( SCIPaddBilinTermQuadratic(scip, cons, x[i], x[i+1], -2.0) );
          SCIP_CALL( SCIPaddSquareCoefQuadratic(scip, cons, v[i], -1.0) );
 
          /* add the constraint to the problem and forget it */
          SCIP_CALL( SCIPaddCons(scip, cons) );
          SCIP_CALL( SCIPreleaseCons(scip, &cons) );
 
          /* add constraint x[i] <= x[i+1], if x[0] < x[N], otherwise add x[i+1] <= x[i] */
          SCIPsnprintf(consname, SCIP_MAXSTRLEN, "xorder(%d)", i);
          SCIP_CALL( SCIPcreateConsBasicLinear(scip, &cons, consname, 0, NULL, NULL, -SCIPinfinity(scip), 0.0) );
          SCIP_CALL( SCIPaddCoefLinear(scip, cons, x[i],   coord[2] < coord[3] ?  1.0 : -1.0) );
          SCIP_CALL( SCIPaddCoefLinear(scip, cons, x[i+1], coord[2] < coord[3] ? -1.0 :  1.0) );
 
          SCIP_CALL( SCIPaddCons(scip, cons) );
          SCIP_CALL( SCIPreleaseCons(scip, &cons) );
       }
    }
 
    /* release intermediate variables */
    for( i = 0; i < n; ++i )
    {
       SCIP_CALL( SCIPreleaseVar(scip, &t[i]) );
       SCIP_CALL( SCIPreleaseVar(scip, &v[i]) );
    }
 
    /* free arrays allocated */
    SCIPfreeBufferArray(scip, &t);
    SCIPfreeBufferArray(scip, &v);
 
    return SCIP_OKAY;
 }
 
 
 /** runs the brachistochrone example*/
 static
 SCIP_RETCODE runBrachistochrone(
    unsigned int          n,                  /**< number of points for discretization */
    SCIP_Real*            coord               /**< array containing [y(0), y(N), x(0), x(N)] */
    )
 {
    SCIP* scip;
    SCIP_VAR** y;
    SCIP_VAR** x;
    unsigned int i;
 
    assert(n >= 2);
 
    SCIP_CALL( SCIPcreate(&scip) );
    SCIP_CALL( SCIPincludeDefaultPlugins(scip) );
 
    SCIPinfoMessage(scip, NULL, "\n");
    SCIPinfoMessage(scip, NULL, "**********************************************\n");
    SCIPinfoMessage(scip, NULL, "* Running Brachistochrone Problem            *\n");
    SCIPinfoMessage(scip, NULL, "* between A=(%g,%g) and B=(%g,%g) with %d points *\n", coord[2], coord[0], coord[3], coord[1], n);
    SCIPinfoMessage(scip, NULL, "**********************************************\n");
    SCIPinfoMessage(scip, NULL, "\n");
 
    /* set gap at which SCIP will stop */
    SCIP_CALL( SCIPsetRealParam(scip, "limits/gap", 0.05) );
 
    SCIP_CALL( setupProblem(scip, n, coord, &x, &y) );
 
    SCIPinfoMessage(scip, NULL, "Original problem:\n");
    SCIP_CALL( SCIPprintOrigProblem(scip, NULL, "cip", FALSE) );
 
    SCIPinfoMessage(scip, NULL, "\nSolving...\n");
    SCIP_CALL( SCIPsolve(scip) );
 
    if( SCIPgetNSols(scip) > 0 )
    {
       SCIPinfoMessage(scip, NULL, "\nSolution:\n");
       SCIP_CALL( SCIPprintSol(scip, SCIPgetBestSol(scip), NULL, FALSE) );
 
       visualizeSolutionGnuplot(scip, SCIPgetBestSol(scip), n, x, y);
    }
 
    /* release problem variables */
    for( i = 0; i < n+1; ++i )
    {
       SCIP_CALL( SCIPreleaseVar(scip, &y[i]) );
       SCIP_CALL( SCIPreleaseVar(scip, &x[i]) );
    }
 
    /* free arrays allocated */
    SCIPfreeMemoryArray(scip, &x);
    SCIPfreeMemoryArray(scip, &y);
 
    SCIP_CALL( SCIPfree(&scip) );
 
    return SCIP_OKAY;
 }
 
 /** main method starting SCIP */
 int main(
    int                   argc,               /**< number of arguments from the shell */
    char**                argv                /**< arguments: number of points and end coordinates y(N), x(N)*/
    )
 {
    SCIP_RETCODE retcode;
 
    /* setting up default problem parameters */
    unsigned int n = 3;
    SCIP_Real coord[4] = { Y_START, Y_END, X_START, X_END };
 
    /* change some parameters if given as arguments */
    if( argc == 4 || argc == 2  )
    {
        char *end1 = NULL;
        char *end2 = NULL;
        char *end3 = NULL;
 
        n = strtol(argv[1], &end1, 10);
        if( argc == 4 )
        {
           coord[1] = strtof(argv[2], &end2);
           coord[3] = strtof(argv[3], &end3);
        }
 
        if( end1 == argv[1] || ( argc == 4 && ( end2 == argv[2] || end3 == argv[3] ) ) )
        {
           fprintf(stderr, "Error: expected real values as arguments.\n");
           return EXIT_FAILURE;
        }
    }
    else if( argc != 1 )
    {
        fprintf(stderr, "Usage: %s [<N> [<y(N)> <x(N)>]]\n", argv[0]);
        return EXIT_FAILURE;
    }
 
    /* check that y(0) > y(N) */
    if( coord[0] <= coord[1] )
    {
       fprintf(stderr, "Error: expected y(N) < 1.0\n");
       return EXIT_FAILURE;
    }
 
    retcode = runBrachistochrone(n, coord);
 
    /* evaluate return code of the SCIP process */
    if( retcode != SCIP_OKAY )
    {
       /* write error back trace */
       SCIPprintError(retcode);
       return EXIT_FAILURE;
    }
 
    return EXIT_SUCCESS;
 }


 SCIP_CALL( SCIPcreateConsNonlinear(scip, cons, name, nlinvars, linvars, lincoefs, nexprtrees, exprtrees,
          nonlincoefs, lhs, rhs,
          TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );
 
  /** creates and captures a nonlinear constraint
  *  this variant takes expression trees as input
  *
  *  @note the constraint gets captured, hence at one point you have to release it using the method SCIPreleaseCons()
  */
 SCIP_RETCODE SCIPcreateConsNonlinear(
    SCIP*                 scip,               /**< SCIP data structure */
    SCIP_CONS**           cons,               /**< pointer to hold the created constraint */
    const char*           name,               /**< name of constraint */
    int                   nlinvars,           /**< number of linear variables in the constraint */
    SCIP_VAR**            linvars,            /**< array with linear variables of constraint entries */
    SCIP_Real*            lincoefs,           /**< array with coefficients of constraint linear entries */
    int                   nexprtrees,         /**< number of expression trees for nonlinear part of constraint */
    SCIP_EXPRTREE**       exprtrees,          /**< expression trees for nonlinear part of constraint */
    SCIP_Real*            nonlincoefs,        /**< coefficients for expression trees for nonlinear part, or NULL if all 1.0 */
    SCIP_Real             lhs,                /**< left hand side of constraint */
    SCIP_Real             rhs,                /**< right hand side of constraint */
    SCIP_Bool             initial,            /**< should the LP relaxation of constraint be in the initial LP?
                                               *   Usually set to TRUE. Set to FALSE for 'lazy constraints'. */
    SCIP_Bool             separate,           /**< should the constraint be separated during LP processing?
                                               *   Usually set to TRUE. */
    SCIP_Bool             enforce,            /**< should the constraint be enforced during node processing?
                                               *   TRUE for model constraints, FALSE for additional, redundant constraints. */
    SCIP_Bool             check,              /**< should the constraint be checked for feasibility?
                                               *   TRUE for model constraints, FALSE for additional, redundant constraints. */
    SCIP_Bool             propagate,          /**< should the constraint be propagated during node processing?
                                               *   Usually set to TRUE. */
    SCIP_Bool             local,              /**< is constraint only valid locally?
                                               *   Usually set to FALSE. Has to be set to TRUE, e.g., for branching constraints. */
    SCIP_Bool             modifiable,         /**< is constraint modifiable (subject to column generation)?
                                               *   Usually set to FALSE. In column generation applications, set to TRUE if pricing
                                               *   adds coefficients to this constraint. */
    SCIP_Bool             dynamic,            /**< is constraint subject to aging?
                                               *   Usually set to FALSE. Set to TRUE for own cuts which
                                               *   are seperated as constraints. */
    SCIP_Bool             removable,          /**< should the relaxation be removed from the LP due to aging or cleanup?
                                               *   Usually set to FALSE. Set to TRUE for 'lazy constraints' and 'user cuts'. */
    SCIP_Bool             stickingatnode      /**< should the constraint always be kept at the node where it was added, even
                                               *   if it may be moved to a more global node?
                                               *   Usually set to FALSE. Set to TRUE to for constraints that represent node data. */
    )
 {
    SCIP_CONSHDLR* conshdlr;
    SCIP_CONSDATA* consdata;
    int i;
 
    assert(linvars  != NULL || nlinvars == 0);
    assert(lincoefs != NULL || nlinvars == 0);
    assert(exprtrees   != NULL || nexprtrees == 0);
    assert(modifiable == FALSE); /* we do not support column generation */
 
    /* find the nonlinear constraint handler */
    conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
    if( conshdlr == NULL )
    {
       SCIPerrorMessage("nonlinear constraint handler not found\n");
       return SCIP_PLUGINNOTFOUND;
    }
 
    /* create constraint data */
    SCIP_CALL( consdataCreateEmpty(scip, &consdata) );
 
    consdata->lhs = lhs;
    consdata->rhs = rhs;
 
    /* create constraint */
    SCIP_CALL( SCIPcreateCons(scip, cons, name, conshdlr, consdata, initial, separate, enforce, check, propagate,
          local, modifiable, dynamic, removable, stickingatnode) );
 
    /* add linear variables */
    SCIP_CALL( consdataEnsureLinearVarsSize(scip, consdata, nlinvars) );
    for( i = 0; i < nlinvars; ++i )
    {
       if( SCIPisZero(scip, lincoefs[i]) )  /*lint !e613*/
          continue;
 
       SCIP_CALL( addLinearCoef(scip, *cons, linvars[i], lincoefs[i]) );  /*lint !e613*/
    }
 
    /* set expression trees */
    SCIP_CALL( consdataSetExprtrees(scip, consdata, nexprtrees, exprtrees, nonlincoefs, TRUE) );
 
    SCIPdebugMsg(scip, "created nonlinear constraint ");
    SCIPdebugPrintCons(scip, *cons, NULL);
 
    return SCIP_OKAY;
 }