#include "svm.hpp"
#include "data.hpp"
#include "utils.hpp"
#include <math.h>
#include <string.h>

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
        Data::iterator iter = (*data)(i);
        while(iter.hasNext()) {
            innr_xw += weights[iter.getIndex()] * iter.next();
        }
        double slack = 1 - (*data)[i] * innr_xw;
        if(slack > 0) {
            _F += slack / (double) data->size();
        }
    }
    return _F;
}

double svm::first_component_oracle_core(Data* data, int given_index, double* weights) const {
    //Sub Gradient For SVM
    if(weights == NULL) weights = m_weights;
    double innr_xw = 0;
    Data::iterator iter = (*data)(given_index);
    while(iter.hasNext()) {
        innr_xw += weights[iter.getIndex()] * iter.next();
    }
    if((*data)[given_index] * innr_xw < 1)
        return -(*data)[given_index];
    else
        return 0.0;
}

int svm::classify(double* sample) const{
    return 1;
}
