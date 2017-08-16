#include "logistic.hpp"
#include "utils.hpp"
#include <math.h>

extern size_t MAX_DIM;
logistic::logistic(double param) {
    m_params = new double;
    *m_params = param;
    m_weights = new double[MAX_DIM];
}

double logistic::zero_component_oracle(Data* data, double* weights) const {
    if(weights == NULL) weights = m_weights;
    double _F = 0.0;
    for(size_t i = 0; i < data->size(); i ++) {
        double innr_yxw = 0.0;
        for(size_t j = 0; j < MAX_DIM; j ++) {
            innr_yxw += (*data)(i, j) * weights[j];
        }
        innr_yxw *= -(*data)[i];
        _F += log(1.0 + exp(innr_yxw));
    }
    return _F / (double) data->size();
}

double logistic::zero_regularizer_oracle(double* weights) const {
    if(weights == NULL) weights = m_weights;
    double l2_norm = comp_l2_norm(weights);
    return *m_params * 0.5 * l2_norm * l2_norm;
}

void logistic::first_component_oracle(Data* data, double* _pF, int given_index, double* weights) const {
    if(weights == NULL) weights = m_weights;
    double exp_yxw = 0.0;
    for(size_t i = 0; i < MAX_DIM; i ++) {
        exp_yxw += (*data)(given_index, i) * weights[i];
        _pF[i] = 0.0;
    }
    exp_yxw = exp(exp_yxw * -(*data)[given_index]);
    for(size_t i = 0; i < MAX_DIM; i ++) {
        _pF[i] = (*data)(given_index, i) * -(*data)[given_index] * exp_yxw / (1 + exp_yxw);
    }
}

void logistic::first_regularizer_oracle(double* _pR, double* weights) const {
    if(weights == NULL) weights = m_weights;
    for(size_t i = 0; i < MAX_DIM; i ++) {
        _pR[i] = weights[i] * (*m_params);
    }
}

void logistic::proximal_regularizer(double* _prox, double step_size) const {
    //Not Applicable
    _prox = NULL;
}

int logistic::classify(double* sample) const{
    return 1;
}
