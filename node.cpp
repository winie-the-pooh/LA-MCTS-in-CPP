//
// Created by lixin on 2021/5/8.
//
#include "node.h"
#include <algorithm>
#include <ctime>

std::default_random_engine Node::generator(time(nullptr));

Node::Node(const std::vector<Eigen::VectorXd> &samples,const unsigned int& dims ,const std::vector<unsigned int> &indices,const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
           const unsigned int &leaf_size/*,const std::string &kernel_type, const std::string &gamma_type*/):samples(samples),dims(dims),indices(indices),
           lb(lb),ub(ub),leaf_size(leaf_size),classifier(samples,indices,dims)
{
    fx_bar = 0;
    for(unsigned int i = 0;i < indices.size();i++)
    {
        const Eigen::VectorXd& sample = samples[indices[i]];
        fx_bar += sample(sample.size()-1);
    }
    fx_bar /= indices.size();
}


bool Node::split() {
    if(indices.size() > leaf_size)
    {
        //Kmeans
        Eigen::VectorXi label = classifier.get_clusters(); //0 for good region, 1 for bad.
//        std::cout << "Kmeans....." << std::endl;
//        std::cout << label.transpose() << std::endl;
        //train SVM
        classifier.get_boundary(label);
//        std::cout << "SVM....." << std::endl;
        //split
        std::vector<unsigned int> left_ind, right_ind;
        for(int i = 0;i < label.size();i++)
        {
            label[i] == 0?left_ind.push_back(indices[i]):right_ind.push_back(indices[i]);
        }
        if(left_ind.empty() || right_ind.empty())
        {
            return false;
        }

        left = new Node(samples,dims,left_ind,lb,ub,leaf_size);
        left->parent = this;
        right = new Node(samples,dims,right_ind,lb,ub,leaf_size);
        right->parent = this;
        return true;
    }
    else
    {
        return false;
    }
}

std::vector<Eigen::VectorXd> Node::raw_propose(const unsigned int &num, const Eigen::VectorXd &center, const double& radius) const {
    unsigned int num1 = num/2;
    std::vector<std::normal_distribution<double> > distribution;
    distribution.reserve(dims);
    for(int i = 0;i < dims;i++)
    {
        distribution.emplace_back(center(i),1 );
    }
    auto normal = [&distribution] (const int& i) {return distribution[i](generator);};

    std::vector<Eigen::VectorXd> proposal;
    proposal.reserve(num);
    for(int i = 0;i < num1;i++)
    {
        proposal.emplace_back(Eigen::VectorXd::NullaryExpr(dims, normal ));
    }

    std::vector<std::uniform_real_distribution<double> > un_distribution;
    un_distribution.reserve(dims);
    for(int i = 0;i < dims;i++)
    {
        un_distribution.emplace_back(center(i)-0.1,center(i)+0.1 );
    }
    auto uniform = [&un_distribution] (const int& i) {return un_distribution[i](generator);};

    for(int i = 0;i < num-num1;i++)
    {
        proposal.emplace_back(Eigen::VectorXd::NullaryExpr(dims, uniform ));
    }

    return proposal;
}

std::vector<Eigen::VectorXd> Node::in_region_filter(const std::vector<Eigen::VectorXd> &proposals) const {
    std::vector<Eigen::VectorXd> res;
    for(int i = 0;i < proposals.size();i++)
    {
        if( (proposals[i].array() >= lb.array()).all() && (proposals[i].array() <= ub.array()).all() )
        {
            res.push_back(proposals[i]);
        }
    }

    return res;
}



std::vector<Eigen::VectorXd> Node::propose_samples(const unsigned int &num, const std::vector<Node *> &p) {
    classifier.train_gpr();
    Eigen::VectorXd mu = Eigen::VectorXd::Zero(dims);
    double denominator = 0;
    double w;
    for(int i = 0;i < indices.size();i++)
    {
        auto& sample = samples[indices[i]];
        w = std::exp(-sample(sample.size()-1)/4);
        mu += sample.head(sample.size()-1)*w;
        denominator += w;
    }
    mu /= denominator;

    unsigned int sample_num = 10000;
    std::vector<Eigen::VectorXd> final_proposal = in_region_filter(raw_propose(sample_num,mu,1.0));
    std::cout << final_proposal.size() << std::endl;


    Eigen::VectorXd score = classifier.gpr_score(final_proposal);
    std::vector<std::pair<Eigen::VectorXd,double> > sort_proposal;
    sort_proposal.reserve(score.size());
    for(int i = 0;i < final_proposal.size();i++)
    {
        sort_proposal.emplace_back(final_proposal[i],score(i));
    }
    auto sorter = [](const std::pair<Eigen::VectorXd,double>& x1,const std::pair<Eigen::VectorXd,double>& x2){
        return x1.second > x2.second;
    };
    std::sort(sort_proposal.begin(),sort_proposal.end(),sorter);

    std::vector<Eigen::VectorXd > return_proposal;
    if(sort_proposal.size() >= num)
    {
        return_proposal.reserve(num);
        for(int i = 0;i < num;i++){
            return_proposal.push_back(sort_proposal[i].first);
        }
    }
    else
    {
        return_proposal.reserve(sort_proposal.size());
        for(int i = 0;i < sort_proposal.size();i++){
            return_proposal.push_back(sort_proposal[i].first);
        }
    }

    std::cout << return_proposal.size() << std::endl;
    return return_proposal;
}
