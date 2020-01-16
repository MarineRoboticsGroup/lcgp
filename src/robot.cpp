#include "robot.h"


namespace {

	float gaussNoise(float mean, float stddev){
		auto dist = std::bind(std::normal_distribution<double>{mean, stddev},
			std::mt19937(std::random_device{}()));
    	return dist();
	}

}

Robot::Robot(Point2d startLoc, int nId, bool isRob) {
	loc = startLoc;
	id = nId;
	isRobot = isRob;
}

Robot::Robot(Point2d startLoc, int nId, bool isRob, float stdDev) {
	loc = startLoc;
	id = nId;
	isRobot = isRob;
	rangeStddev = stdDev;
}

Robot::Robot(){}

Robot::~Robot(){}

float Robot::distTo(Robot rob){

	Point2d p2 = rob.getCurrLoc();
	int id2 = rob.getId();
    float d = Point2d(loc.getX() - p2.getX(), 
        loc.getY() - p2.getY()).norm();
    
    d += gaussNoise(0, rangeStddev);

    return d;
}

void Robot::move(float x, float y) {
	if (!isRobot){ return; }

	Point2d currLoc = getCurrLoc();
	float netDist = sqrt(x*x + y*y);
	if (netDist > controlSat)
	{
		x /= netDist;
		y /= netDist;
		x *= controlSat;
		y *= controlSat;	
	}

	float xNew = currLoc.getX() + x;
	float yNew = currLoc.getY() + y;
	Point2d newLoc = Point2d(xNew, yNew);
	loc = newLoc;
}
