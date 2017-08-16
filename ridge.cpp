#include "ridge.hpp"
#include "utils.hpp"
extern size_t MAX_DIM;
ridge::ridge(double param) {
    m_params = new double;
    *m_params = param;
    m_weights = new double[MAX_DIM];
}

double ridge::zero_component_oracle(Data* data, double* weights) const {
    if(weights == NULL) weights = m_weights;
    double _F = 0.0;
    for(size_t i = 0; i < data->size(); i ++) {
        double _inner_xw = 0;
        for(size_t j = 0; j < MAX_DIM; j ++) {
            _inner_xw += (*data)(i, j) * weights[j];
        }
        _F += (_inner_xw - (*data)[i]) * (_inner_xw - (*data)[i])
            / (double) data->size();
    }
    return _F;
}

double ridge::zero_regularizer_oracle(double* weights) const {
    if(weights == NULL) weights = m_weights;
    double l2_norm = comp_l2_norm(weights);
    return *m_params * 0.5 * l2_norm * l2_norm;
}

void ridge::first_component_oracle(Data* data, double* _pF, int given_index, double* weights) const {
    if(weights == NULL) weights = m_weights;
    double _loss = 0, _inner_xw = 0;
    for(size_t i = 0; i < MAX_DIM; i ++) {
        _inner_xw += (*data)(given_index, i) * weights[i];
        _pF[i] = 0.0;
    }
    _loss = _inner_xw - (*data)[given_index];
    for(size_t i = 0; i < MAX_DIM; i ++) {
        _pF[i] += 2.0 * _loss * (*data)(given_index, i);
    }
}

void ridge::first_regularizer_oracle(double* _pR, double* weights) const {
    if(weights == NULL) weights = m_weights;
    for(size_t i = 0; i < MAX_DIM; i ++) {
        _pR[i] = (*m_params) * weights[i];
    }
}

void ridge::proximal_regularizer(double* _prox, double step_size) const {
    //Not Applicable
    _prox = NULL;
}

int ridge::classify(double* sample) const{
    return 1;
}
