#ifndef PLOT_H_
#define PLOT_H_

#include "matplotlibcpp.h"
#include "geom.h"
#include <vector>

class Plot
{
public:	

	/**
	 * @brief      Plots the given circle
	 *
	 * @param[in]  c     circle to plot
	 */
	void plotCircle(Circle c);


	/**
	 * @brief      Plots the given circle
	 *
	 * @param[in]  p     center of circle
	 * @param[in]  r     radius of circle
	 * @param[in]  col   The color to be used
	 */
	void plotCircle(Point2d p, float r, int col);

	
	/**
	 * @brief      Plots a line defined by two endpoints
	 *
	 */
	void plotLine(Point2d p1, Point2d p2);
	
	/**
	 * @brief      Plots a given point
	 *
	 */
	void plotPoint(Point2d p, int col);
	

	void showPlot();
	void animation();
	void setAxis(float xlim, float ylim);
	void setAxisEqual();

private:
	std::vector<std::string> colors = {"b", "g", "r", "c", "m", "y", "k"};

};


#endif
