//
// Created by lixin on 2021/5/17.
//

#ifndef LA_MCTS_CLASSIFIER_H
#define LA_MCTS_CLASSIFIER_H
#include <thundersvm/model/svc.h>
#include <thundersvm/util/metric.h>
#include <GaussianProcess.h>
#include <random>

typedef gpr::GaussianKernel<double> KernelType;
typedef ::std::shared_ptr<KernelType> KernelTypePointer;
typedef gpr::GaussianProcess<double> GaussianProcessType;
typedef ::std::shared_ptr<GaussianProcessType> GaussianProcessTypePointer;
typedef GaussianProcessType::VectorType VectorType;
typedef GaussianProcessType::MatrixType MatrixType;

class Classifier{

    unsigned int dims;
    const ::std::vector<Eigen::VectorXd>& samples;
    ::std::vector<unsigned int> indices;
    ::std::shared_ptr<SvmModel> model;
    GaussianProcessTypePointer gp;

public:
    static std::uniform_int_distribution<unsigned> u;
    static std::default_random_engine generator;

    Classifier(const ::std::vector<Eigen::VectorXd>& samples,const ::std::vector<unsigned int>& indices,const unsigned int& dims);

    Eigen::VectorXi get_clusters();
    Eigen::VectorXi in_boundary(const ::std::vector<Eigen::VectorXd>& new_samples); // 0 for in.
    void get_boundary(const Eigen::VectorXi& label);
    double normal_pdf(const double& x, const double& m, const double& s) const;
    Eigen::VectorXd gpr_score(const ::std::vector<Eigen::VectorXd>& new_samples);
    void train_gpr();

};


#endif //LA_MCTS_CLASSIFIER_H
