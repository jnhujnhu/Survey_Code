#include "least_square.hpp"
extern size_t MAX_DIM;
least_square::least_square(double param, int regular) {
    m_regularizer = regular;
    m_params = new double;
    *m_params = param;
    m_weights = new double[MAX_DIM];
}

double least_square::zero_component_oracle(Data* data, double* weights) const {
    if(weights == NULL) weights = m_weights;
    double _F = 0.0;
    for(size_t i = 0; i < data->size(); i ++) {
        double _inner_xw = 0;
        for(size_t j = 0; j < MAX_DIM; j ++) {
            _inner_xw += (*data)(i, j) * weights[j];
        }
        _F += (_inner_xw - (*data)[i]) * (_inner_xw - (*data)[i]);
    }
    _F /= (double) data->size();
    return _F;
}

void least_square::first_component_oracle(Data* data, double* _pF, int given_index, double* weights) const {
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

int least_square::classify(double* sample) const{
    return 1;
}
