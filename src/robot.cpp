#include "robot.h"


Robot::Robot(Point2d startLoc, int nId) {
	loc = startLoc;
	id = nId;
}

Robot::Robot(){}

Robot::~Robot(){}

float Robot::distToRob(Robot rob){
	Point2d p2 = rob.getCurrLoc();
	int id2 = rob.getRobotId();
    float d = Point2d(loc.getX() - p2.getX(), 
        loc.getY() - p2.getY()).norm();
    // ranges.at(id2) =  d;
    return d;
}

void Robot::move(float x, float y) {
	Point2d currLoc = getCurrLoc();
	float netDist = sqrt(x*x + y*y);
	if (netDist > controlSat)
	{
		netDist *= controlSat;
	}

	float xNew = currLoc.getX() + x/netDist;
	float yNew = currLoc.getY() + y/netDist;
	Point2d newLoc = Point2d(xNew, yNew);
	loc = newLoc;
}
