#include "logistic.hpp"
#include "utils.hpp"
#include <math.h>
#include <string.h>

extern size_t MAX_DIM;
logistic::logistic(double param, int _regularizer) {
    m_regularizer = _regularizer;
    m_params = new double;
    *m_params = param;
    m_weights = new double[MAX_DIM];
}

double logistic::zero_component_oracle(Data* data, double* weights) const {
    if(weights == NULL) weights = m_weights;
    double _F = 0.0;
    for(size_t i = 0; i < data->size(); i ++) {
        double innr_yxw = 0.0;
        Data::iterator iter = (*data)(i);
        while(iter.hasNext()) {
            innr_yxw += weights[iter.getIndex()] * iter.next();
        }
        innr_yxw *= -(*data)[i];
        _F += log(1.0 + exp(innr_yxw));
    }
    return _F / (double) data->size();
}

double logistic::first_component_oracle_core(Data* data, int given_index, double* weights) const {
    if(weights == NULL) weights = m_weights;
    double sigmoid = 0.0;
    Data::iterator iter = (*data)(given_index);
    while(iter.hasNext()){
        sigmoid += weights[iter.getIndex()] * iter.next();
    }
    sigmoid = exp(sigmoid * -(*data)[given_index]);
    sigmoid = -(*data)[given_index] * sigmoid /  (1 + sigmoid);
    return sigmoid;
}

int logistic::classify(double* sample) const{
    return 1;
}
