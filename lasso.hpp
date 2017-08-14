#ifndef LASSO_H
#define LASSO_H
#include "blackbox.hpp"

class lasso: public blackbox {
public:
    lasso(double param);
    int classify(double* sample) const override;
    double zero_oracle(Data* data, double* weights = NULL) const override;
    double* first_oracle(Data* data, bool is_stochastic = false
        , std::default_random_engine* generator = NULL
        , std::uniform_int_distribution<int>* distribution = NULL
        , double* weights = NULL) const override;
    double* first_oracle(Data* data, int given_index, double* weights = NULL) const override;
};

#endif
