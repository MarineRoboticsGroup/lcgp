#ifndef GEOM_H_
#define GEOM_H_ 

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <vector>

 
/*
 * Find the intersection point(s) of two circles,
 * when their centers and radiuses are given (2D).
 */
 
class Point2d{
public:
    Point2d() {}
    Point2d(float x, float y)
        : X(x), Y(y) {}
     
    float getX() const { return X; }
    float getY() const { return Y; }
     
    /**
     * Returns the norm of this std::vector.
     *
     * @return     the norm
     */
    float norm() const {
        return sqrt( X * X + Y * Y );
    }

     
private:
    float X;
    float Y;
};
 
class Circle{
public:
    /**
     * @brief      Constructs a new instance.
     *
     * @param      R     - radius
     * @param      C     - center
     */
    Circle(float R, Point2d& C) 
        : r(R), c(C) {}
         
    /**
     * @brief      Constructs a new instance.
     *
     * @param      R     - radius
     * @param      X     - center's x coordinate
     * @param      Y     - center's y coordinate
     */
    Circle(float R, float X, float Y) 
        : r(R), c(X, Y) {}    
     
    Point2d getC() const { return c; }
    float getR() const { return r; }
     
    /**
     * @brief      Calculates the intersection of current circle and given
     *             circle
     *
     * @param[in]  C2    other circle
     *
     * @return     vector of intersection points
     */
    std::vector<Point2d> intersect(const Circle& C2) {
        std::vector<Point2d> intersects;

        float r1 = r, r2 = C2.r; 
        Point2d c1 = c, c2 = C2.c;
        
        // distance between the centers
        float d = Point2d(c1.getX() - c2.getX(), 
                c1.getY() - c2.getY()).norm();
         
        // find number of solutions
        if(d >r1 + r2) // circles are too far apart, no solution(s)
        {
            std::cout << "Circles are too far apart\n";
        }
        else if(d == 0 && r1 == r2) // circles coincide
        {
            std::cout << "Circles coincide\n";
        }
        // one circle contains the other
        else if(d + std::min(r1, r2) < std::max(r, r2))
        {
            std::cout << "One circle contains the other\n";
        }
        else
        {
            float a = (r1*r1 - r2*r2 + d*d)/ (2.0*d);
            float h = sqrt(r1*r1 - a*a);
             
            // find p2
            Point2d p2( c1.getX() + (a * (c2.getX() - c1.getX())) / d,
                    c1.getY() + (a * (c2.getY() - c1.getY())) / d);
             
            // find intersection points p3
            intersects.push_back(Point2d( p2.getX() + (h * (c2.getY() - c1.getY())/ d),
                    p2.getY() - (h * (c2.getX() - c1.getX())/ d)));
            
            // if there are 2 intersects
            if(d != r1 + r2){
                intersects.push_back(Point2d( p2.getX() - (h * (c2.getY() - c1.getY())/ d),
                        p2.getY() + (h * (c2.getX() - c1.getX())/ d))
                );
            }
        }
        return intersects;  
    }
     
private:
    // radius
    float r;
    // center
    Point2d c;
     
};

#endif
