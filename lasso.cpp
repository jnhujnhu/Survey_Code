#include "lasso.hpp"

extern size_t MAX_DIM;
lasso::lasso(double param) {
    m_params = new double;
    *m_params = param;
    m_weights = new double[MAX_DIM];
}

double lasso::zero_component_oracle(Data* data, double* weights) const {
    return 1.0;
}

double lasso::zero_regularizer_oracle(double* weights) const {
    return 1.0;
}

void lasso::first_component_oracle(Data* data, double* _pF, int given_index, double* weights) const {
}

void lasso::first_regularizer_oracle(double* _pR, double* weights) const {
}

void lasso::proximal_regularizer(double* _prox, double step_size) const {
}

int lasso::classify(double* sample) const{
    return 1;
}
