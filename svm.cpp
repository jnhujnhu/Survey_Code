#include "svm.hpp"
#include "data.hpp"
#include "utils.hpp"
#include <math.h>

extern size_t MAX_DIM;
svm::svm(double param, int regular) {
    m_regularizer = regular;
    m_params = new double;
    *m_params = param;
    m_weights = new double[MAX_DIM];
}

double svm::zero_component_oracle(Data* data, double* weights) const {
    if(weights == NULL) weights = m_weights;
    double _F = 0.0;
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

void svm::first_component_oracle(Data* data, double* _pF, int given_index, double* weights) const {
    //Sub Gradient For SVM
    if(weights == NULL) weights = m_weights;
    double innr_xw = 0;
    for(size_t i = 0; i < MAX_DIM; i ++) {
        _pF[i] = 0.0;
        innr_xw += (*data)(given_index, i) * weights[i];
    }
    if((*data)[given_index] * innr_xw < 1) {
        for(size_t i = 0; i < MAX_DIM; i ++) {
            _pF[i] -= (*data)[given_index] * (*data)(given_index, i);
        }
    }
}

int svm::classify(double* sample) const{
    return 1;
}
