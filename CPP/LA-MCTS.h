//
// Created by lixin on 2021/5/6.
//

#ifndef LA_MCTS_LA_MCTS_H
#define LA_MCTS_LA_MCTS_H
#include <limits>
#include "functions.h"
#include "node.h"

class MCTS{
    unsigned int dims;
    std::vector<Eigen::VectorXd> samples;
    Node* root = nullptr;
    double Cp;
    Eigen::VectorXd lb;
    Eigen::VectorXd ub;
    unsigned int ninits;
    Function* f;
    double cur_best_value = std::numeric_limits<double >::max();
    Eigen::VectorXd cur_best_sample;

    unsigned int leaf_size;
    std::string kernel_type;
    std::string gamma_type;

public:
    MCTS(const Eigen::VectorXd& lb,
         const Eigen::VectorXd& ub,
         const unsigned int& dims,
         const unsigned int& ninits,
         Function* const& f,
         const double& Cp,
         const unsigned int& leaf_size,
         const std::string& kernel_type,
         const std::string& gamma_type);

    void deletetree();
    void expandtree();
    Node* select(std::vector<Node*>& path);
    void search(const unsigned int& iterations);
    double collect_sample(const Eigen::VectorXd& sample);
    void init_train();


};

#endif //LA_MCTS_LA_MCTS_H
