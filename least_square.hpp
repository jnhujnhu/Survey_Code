#ifndef RIDGE_H
#define RIDGE_H
#include "blackbox.hpp"
#include "regularizer.hpp"

class least_square: public blackbox {
public:
    least_square(double param, int regular = regularizer::L2);
    int classify(double* sample) const override;
    double zero_component_oracle(Data* data, double* weights = NULL) const override;
    void first_component_oracle(Data* data, double* _pF, int given_index, double* weights = NULL) const override;
};

#endif