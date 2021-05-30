//
// Created by lixin on 2021/5/8.
//

#ifndef LA_MCTS_NODE_H
#define LA_MCTS_NODE_H

#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include "Classifier.h"
#include <random>

class Node{
    Node* parent = this;
    Node* left = nullptr;
    Node* right = nullptr;

    const std::vector<Eigen::VectorXd >& samples;
    const Eigen::VectorXd& lb;
    const Eigen::VectorXd& ub;
    std::vector<unsigned int> indices;
    double fx_bar;
    unsigned int dims;
    unsigned int leaf_size;
    Classifier classifier;
public:
    static std::default_random_engine generator;

    Node(const std::vector<Eigen::VectorXd >& samples,const unsigned int& dims, const std::vector<unsigned int>& indices,
         const Eigen::VectorXd& lb, const Eigen::VectorXd& ub, const unsigned int& leaf_size/*,
         const std::string& kernel_type, const std::string& gamma_type*/);
    Node* getleft(){ return left;}
    Node* getright(){ return right;}

    double get_uct(const double& Cp){
        if(indices.empty())
        {
            return std::numeric_limits<double>::max();
        }
        else
        {
            return -fx_bar + 2*Cp*std::sqrt( 2* std::pow(parent->indices.size(), 0.5) / indices.size() );
        }
    }
    bool split();
    std::vector<Eigen::VectorXd> raw_propose(const unsigned int& num,const Eigen::VectorXd& center, const double& radius) const;
    std::vector<Eigen::VectorXd> in_region_filter(const std::vector<Eigen::VectorXd>& proposals) const;
    std::vector<Eigen::VectorXd> propose_samples(const unsigned int& num,const std::vector<Node*>& p);
};

#endif //LA_MCTS_NODE_H
