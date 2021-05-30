//
// Created by lixin on 2021/5/6.
//
#include <iostream>
#include <stack>
#include "LA-MCTS.h"

MCTS::MCTS(const Eigen::VectorXd &lb, const Eigen::VectorXd &ub, const unsigned int &dims, const unsigned int &ninits,
           Function* const &f, const double &Cp, const unsigned int&leaf_size, const std::string &kernel_type,
           const std::string &gamma_type):lb(lb),ub(ub),dims(dims),ninits(ninits),f(f),Cp(Cp),leaf_size(leaf_size),
           kernel_type(kernel_type),gamma_type(gamma_type){

    init_train();
}

void MCTS::init_train() {
    Eigen::MatrixXd S = Eigen::MatrixXd::Random(ninits,dims);
    S = (((S.array() + 1)/2).rowwise()*(ub-lb).transpose().array()).rowwise()+lb.transpose().array();
    for(int i = 0;i < ninits;i++)
    {
        collect_sample( S.row(i) );
    }
}
double MCTS::collect_sample(const Eigen::VectorXd &sample) {
    double val = (*f)(sample);
    if(val < cur_best_value)
    {
        cur_best_value = val;
        cur_best_sample = sample;
    }

    Eigen::VectorXd temp(sample.size() + 1);
    temp << sample,val;
    samples.push_back(temp);

    return val;
}

void MCTS::deletetree() {
    if(root)
    {
        std::stack<Node*> s;
        s.push(root);
        Node* cur;
        while(!s.empty())
        {
            cur = s.top();
            s.pop();
            if(cur->getleft()){s.push(cur->getleft());}
            if(cur->getright()){s.push(cur->getright());}
            delete cur;
        }
    }
}

void MCTS::expandtree() {
    deletetree();
    std::vector<unsigned int> ind;
    ind.reserve(samples.size());
    for(int i = 0;i < samples.size();i++)
    {
        ind.push_back(i);
    }
    root = new Node(samples,dims,ind, this->lb, this->ub,this->leaf_size/*,this->kernel_type,this->gamma_type*/);
    std::stack<Node*> s;
    s.push(root);
    Node* cur;
    while(!s.empty())
    {
        cur = s.top();
        s.pop();
        if(cur->split())
        {
            s.push(cur->getright());
            s.push(cur->getleft());
        }
    }
}

Node * MCTS::select(std::vector<Node *> &path) {
    Node* cur = root;
    path.clear();
    path.push_back(cur);
    Node *l,*r;
    while( (l = cur->getleft()) && (r = cur->getright()) )
    {
        if(l->get_uct(Cp) >= r->get_uct(Cp))
        {
            cur = l;
        }
        else
        {
            cur = r;
        }
        path.push_back(cur);
    }
    return cur;
}


void MCTS::search(const unsigned int &iterations) {
    for(unsigned int i = ninits;i < iterations;i++)
    {
        std::cout << std::endl;
        std::cout << "==========" << std::endl;
        std::cout << "Sample:" << i + 1 << std::endl;
        std::cout << "==========" << std::endl;
        expandtree();
        std::vector<Node*> path;
        Node* leaf = select(path);
        auto new_samples = leaf->propose_samples(1,path);
        for(auto it = new_samples.begin();it != new_samples.end();it++)
        {
            collect_sample(*it);
        }
        std::cout << "total samples:" << samples.size() << std::endl;
        std::cout << "current best f(x):" << cur_best_value << std::endl;
        std::cout << "current best x:" << cur_best_sample.transpose() << std::endl;
    }
}