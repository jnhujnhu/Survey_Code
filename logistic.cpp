#include "logistic.hpp"
#include "utils.hpp"
#include <math.h>

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

void logistic::first_component_oracle(Data* data, double* _pF, int given_index, double* weights) const {
    if(weights == NULL) weights = m_weights;
    double exp_yxw = 0.0;
    Data::iterator iter = (*data)(given_index);
    while(iter.hasNext()){
        exp_yxw += weights[iter.getIndex()] * iter.next();
    }
    exp_yxw = exp(exp_yxw * -(*data)[given_index]);

    iter.reset(given_index);
    while(iter.hasNext()) {
        // Prevent malposition for index.
        size_t index = iter.getIndex();
        _pF[index] = iter.next() * -(*data)[given_index] * exp_yxw / (1 + exp_yxw);
    }
}

int logistic::classify(double* sample) const{
    return 1;
}
