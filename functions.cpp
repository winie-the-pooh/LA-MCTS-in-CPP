//
// Created by lixin on 2021/5/8.
//
#include <sstream>
#include <cmath>
#include "functions.h"
#include "Macros.h"

Ackley::Ackley(const unsigned &dims):
Function(10,Eigen::VectorXd::Constant(dims,-5),Eigen::VectorXd::Constant(dims,10),0,1,10,40,"rbf","auto")
{
    std::stringstream ss;
    ss << "Ackley" << dims;
    tracker = Tracker(ss.str());
}


double Ackley::operator()(const Eigen::VectorXd& x) {
    counter += 1;
    Assert(x.size() == dims,"invalid input dimension.");
    bool valid = true;
    int temp = ((x.array() > ub.array()) + (x.array() < lb.array())).sum();
    Assert(temp == 0, "x out of searching region.");
    double result = (-20*std::exp(-0.2 * std::sqrt( x.dot(x) /x.size() )) -std::exp( (2*M_PI*x).array().cos().sum() /x.size()) + 20 + M_E);
    tracker.track( result );
    return result;
}