//
// Created by lixin on 2021/5/17.
//
#include "Classifier.h"
#include "KMeansRex/KMeansRexCore.cpp"
//#include <Kernel.h>
#include <ctime>

std::uniform_int_distribution<unsigned> Classifier::u(1,100000);
std::default_random_engine Classifier::generator(time(nullptr));

Classifier::Classifier(const std::vector<Eigen::VectorXd>& samples,const std::vector<unsigned int>& indices,const unsigned int& dims):
samples(samples),indices(indices),dims(dims)
{
    KernelTypePointer k(new KernelType(1000));
    gp = std::make_shared<GaussianProcessType>(k);
    gp->SetSigma(0);

    model = std::make_shared<SVC>();
}

Eigen::VectorXi Classifier::get_clusters() {
    int K = 2;
    int n_iters = 1000;
    int seed = u(generator);

    Eigen::ArrayXXd data(indices.size(),dims+1);
    for(int i = 0;i < indices.size();i++)
    {
        data << samples[indices[i]].transpose();
    }

    Eigen::ArrayXXd mu = Eigen::ArrayXXd::Zero(K, dims+1);
    Eigen::ArrayXd z = Eigen::ArrayXd::Zero(indices.size());

    RunKMeans(
            data.data(),
            indices.size(), dims+1, K,
            n_iters, seed, "plusplus",
            mu.data(),
            z.data());

    if(mu(0,dims) < mu(1,dims))
    {
        return z.round().cast<int>();
    }
    else{
        return (1-z).round().cast<int>();
    }
}

Eigen::VectorXi Classifier::in_boundary(const std::vector<Eigen::VectorXd> &new_samples) {
    DataSet dataset;
    Eigen::ArrayXXf data(new_samples.size(),dims);
    for(int i = 0;i < new_samples.size();i++)
    {
        data << (new_samples[i].transpose()).cast<float>();
    }
    dataset.load_from_dense(new_samples.size(),dims,data.data(), nullptr);
    std::vector<double> predict_y = model->predict(dataset.instances(), 100);
    Eigen::VectorXi res(predict_y.size());
    for(int i = 0;i < predict_y.size();i++)
    {
        res << predict_y[i];
    }
    return res;
}

void Classifier::get_boundary(const Eigen::VectorXi &label) {
    SvmParam param;
    param.gamma = 0.5;
    param.C = 100;
    param.kernel_type = SvmParam::RBF;
    param.epsilon = 3.;//indices.size()*3/10;

    Eigen::ArrayXXf data(indices.size(),dims);
    for(int i = 0;i < indices.size();i++)
    {
        auto& sample = samples[indices[i]];
        data << (sample.head(dims).transpose()).cast<float>();
    }

    Eigen::VectorXf l = label.cast<float>();
    DataSet train_dataset;
    train_dataset.load_from_dense(indices.size(),dims,data.data(),l.data());
//    std::cout << "training." << std::endl;
    model->train(train_dataset, param);
}

double Classifier::normal_pdf(const double& x, const double& m, const double& s) const
{
    static const double inv_sqrt_2pi = 0.3989422804014327;
    double a = (x - m) / s;

    return inv_sqrt_2pi / s * std::exp(-0.5f * a * a);
}



Eigen::VectorXd Classifier::gpr_score(const std::vector<Eigen::VectorXd> &new_samples) {

    // compute expected improvement.
    double trade_off = 0.001;
    Eigen::VectorXd mu(samples.size());
    for(int i = 0;i < samples.size();i++)
    {
        mu << gp->Predict(samples[i].head(dims) );
    }
    double mu_min = mu.minCoeff();
    std::cout << mu_min << std::endl;

    double sigma = 0;
    double Mu,imp,Z;
    Eigen::VectorXd res(new_samples.size());
    for(int i = 0;i < new_samples.size();i++)
    {
        sigma = gp->GetCredibleInterval(new_samples[i]);
        Mu = gp->Predict(new_samples[i])(0);
        imp = mu_min - Mu - trade_off;
        if(sigma != 0.0)
        {
            Z = imp / sigma;
            res << imp * 0.5 * erfc(-Z * M_SQRT1_2) + sigma * normal_pdf(Z,0,1);
        }
        else
        {
            res << 0;
        }
    }
    return res;
}

void Classifier::train_gpr() {
    for(int i = 0;i < indices.size();i++)
    {
        auto& sample = samples[indices[i]];
        gp->AddSample(sample.head(dims),sample.tail(1));
    }
    gp->Initialize();
}
