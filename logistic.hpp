#ifndef LOGISTIC_H
#define LOGISTIC_H
#include "blackbox.hpp"
#include "regularizer.hpp"

class logistic: public blackbox {
public:
    logistic(double param, int _regularizer = regularizer::L2);
    int classify(double* sample) const override;
    double zero_component_oracle(Data* data, double* weights = NULL) const override;
    double first_component_oracle_core(Data* data, int given_index, double* weights = NULL) const override;
};

#endif
