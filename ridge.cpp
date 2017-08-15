#include "ridge.hpp"

extern size_t MAX_DIM;
ridge::ridge(double param) {
    m_params = new double;
    *m_params = param;
    m_weights = new double[MAX_DIM];
}

double ridge::zero_oracle(Data* data, double* weights) const {
    if(weights == NULL) weights = m_weights;
    double _F = (*m_params) / 2.0, innr_w = 0.0;
    for(size_t i = 0; i < MAX_DIM; i ++) {
        innr_w += weights[i] * weights[i];
    }
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

void ridge::first_oracle(Data* data, double* _pF, int given_index, double* weights) const {
    if(weights == NULL) weights = m_weights;
    for(size_t i = 0; i < MAX_DIM; i ++) {
        _pF[i] = (*m_params) * weights[i];
    }
    double _loss = 0, _inner_xw = 0;
    for(size_t i = 0; i < MAX_DIM; i ++) {
        _inner_xw += (*data)(given_index, i) * weights[i];
    }
    _loss = _inner_xw - (*data)[given_index];
    for(size_t i = 0; i < MAX_DIM; i ++) {
        _pF[i] += 2.0 * _loss * (*data)(given_index, i);
    }
}

int ridge::classify(double* sample) const{
    return 1;
}
