#include "plot.h"
namespace plt = matplotlibcpp;



// Geometric Plots

void Plot::plotCircle(Circle c){
	auto center = c.getC();
	auto r = c.getR();
	float x0 = center.getX(), y0 = center.getY();

	const unsigned n = 100;
	std::vector<float> x(n), y(n);

	for (unsigned i = 0; i < n; ++i) {
		x[i] = x0 + r*sin(2 * M_PI * i / n);
		y[i] = y0 + r*cos(2 * M_PI * i / n);
	}

	plt::scatter(x, y, 10);
}

void Plot::plotCircle(Point2d p, float r, int col){
	float x0 = p.getX(), y0 = p.getY();

	const unsigned n = 500;
	std::vector<float> x(n), y(n);

	for (unsigned i = 0; i < n; ++i) {
		x[i] = x0 + r*sin(2 * M_PI * i / n);
		y[i] = y0 + r*cos(2 * M_PI * i / n);
	}
	plt::plot(x, y, {{"color", colors[col]}});
}

void Plot::plotLine(Point2d p1, Point2d p2){
	int n = 100;
	float x1 = p1.getX(), y1 = p1.getY(), x2 = p2.getX(), y2 = p2.getY();
	std::vector<float> x(n), y(n);
	for(int i=0; i<n; ++i) {
		x.at(i) = x1*(((float)i)/n)+x2*(1.0-((float)i)/n);
		y.at(i) = y1*(((float)i)/n)+y2*(1.0-((float)i)/n);
	}
	plt::plot(x, y,"--");
}


void Plot::plotPoint(Point2d p, int col){
	std::vector<double> x(1), y(1); 
	x[0] = p.getX(), y[0] = p.getY();
	plt::scatter(x, y, 50, {{"color", colors[col]}});
}

void Plot::plotPoint(Point2d p, int col, float alpha){
	std::vector<double> x(1), y(1); 
	x[0] = p.getX(), y[0] = p.getY();
	plt::scatter(x, y, 50, {{"color", colors[col]}, {"alpha", std::to_string(alpha)}, {"marker", "^"}});
}

// Plot Controls

void Plot::showPlot(){
	plt::show();
}

void Plot::animation(){
	plt::pause(0.25); 
	plt::clf();
}

void Plot::setAxis(float xlim, float ylim){
	plt::xlim(-xlim, xlim); 
	plt::ylim(-ylim, ylim);
}

void Plot::setAxisEqual(){
	plt::axis("equal");
}