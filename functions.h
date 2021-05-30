//
// Created by lixin on 2021/5/8.
//

#ifndef LA_MCTS_FUNCTIONS_H
#define LA_MCTS_FUNCTIONS_H

#include <string>
#include <Eigen/Dense>
#include "tracker.h"

class Function{
protected:
    unsigned dims;
    Eigen::VectorXd lb;
    Eigen::VectorXd ub;
    unsigned counter;

    double Cp;
    unsigned leaf_size;
    unsigned ninits;
    std::string kernel_type;
    std::string gamma_type;
public:
    Function(
    const unsigned& dims,
    const Eigen::VectorXd& lb,
    const Eigen::VectorXd& ub,
    const unsigned& counter,
    const double& Cp,
    const unsigned& leaf_size,
    const unsigned& ninits,
    const std::string& kernel_type,
    const std::string& gamma_type
    ):dims(dims),lb(lb),ub(ub),counter(counter),Cp(Cp),leaf_size(leaf_size),ninits(ninits),kernel_type(kernel_type),gamma_type(gamma_type){};
    virtual double operator()(const Eigen::VectorXd& x) = 0;

    virtual Eigen::VectorXd getlb() const {return lb;}
    virtual Eigen::VectorXd getub() const {return ub;}
    virtual unsigned getdims() const { return dims;}
    virtual unsigned getninits() const { return ninits;}
    virtual double getCp() const { return Cp;}
    virtual unsigned getleaf_size() const { return leaf_size;}
    virtual std::string getkernel_type() const { return kernel_type;}
    virtual std::string getgamma_type() const { return gamma_type;}
};


class Ackley:public Function{
    Tracker tracker;
public:
    Ackley(const unsigned& dims);
    double operator()(const Eigen::VectorXd& x) override ;
};


#endif //LA_MCTS_FUNCTIONS_H
