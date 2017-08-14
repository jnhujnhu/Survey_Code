#include "svm.hpp"
#include "data.hpp"
#include <math.h>

extern size_t MAX_DIM;
svm::svm(double param) {
    m_params = new double;
    *m_params = param;
    m_weights = new double[MAX_DIM];
}

double svm::zero_oracle(Data* data, double* weights) const {
    if(weights == NULL) weights = m_weights;
    double _F = *m_params / 2.0, innr_w = 0.0;
    for(size_t i = 0; i < MAX_DIM; i ++) {
        innr_w += weights[i] * weights[i];
    }
    _F *= innr_w;
    for(size_t i = 0; i < data->size(); i ++) {
        double innr_xw = 0;
        for(size_t j = 0; j < MAX_DIM; j ++) {
            innr_xw += (*data)(i, j) * weights[j];
        }
        double slack = 1 - (*data)[i] * innr_xw;
        if(slack > 0) {
            _F += slack / (double) data->size();
        }
    }
    return _F;
}

double* svm::first_oracle(Data* data, int given_index, double* weights) const{
    if(weights == NULL) weights = m_weights;
    double* _pF = new double[MAX_DIM];
    for(size_t i = 0; i < MAX_DIM; i ++) {
        _pF[i] = weights[i] * (*m_params);
    }
    double innr_xw = 0;
    for(size_t i = 0; i < MAX_DIM; i ++) {
        innr_xw += (*data)(given_index, i) * weights[i];
    }
    if((*data)[given_index] * innr_xw < 1) {
        for(size_t i = 0; i < MAX_DIM; i ++) {
            _pF[i] -= (*data)[given_index] * (*data)(given_index, i);
        }
    }
    return _pF;
}

double* svm::first_oracle(Data* data, bool is_stochastic
    , std::default_random_engine* generator
    , std::uniform_int_distribution<int>* distribution, double* weights) const {
    if(weights == NULL) weights = m_weights;
    if(is_stochastic) {
        int rand_samp = (*distribution)(*generator);
        return first_oracle(data, rand_samp, weights);
    }
    else {
        double* _pF = new double[MAX_DIM];
        for(size_t i = 0; i < data->size(); i ++) {
            double* _pf = first_oracle(data, i, weights);
            for(size_t j = 0; j < MAX_DIM; j ++)
                _pF[j] += _pf[j];
        }
        for(size_t j = 0; j < MAX_DIM; j ++)
            _pF[j] /= data->size();
        return _pF;
    }
}

int svm::classify(double* sample) const{
    return 1;
}
