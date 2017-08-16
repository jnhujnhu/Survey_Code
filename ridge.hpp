#ifndef RIDGE_H
#define RIDGE_H
#include "blackbox.hpp"

class ridge: public blackbox {
public:
    ridge(double param);
    int classify(double* sample) const override;
    double zero_component_oracle(Data* data, double* weights = NULL) const override;
    double zero_regularizer_oracle(double* weights = NULL) const override;
    void first_component_oracle(Data* data, double* _pF, int given_index, double* weights = NULL) const override;
    void first_regularizer_oracle(double* _pR, double* weights = NULL) const override;
    void proximal_regularizer(double* _prox, double step_size) const override;
};

#endif
