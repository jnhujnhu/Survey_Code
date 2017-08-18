#include "blackbox.hpp"
#include "regularizer.hpp"

blackbox::~blackbox() {
    delete[] m_weights;
    delete[] m_params;
}

double blackbox::zero_oracle(Data* data, double* weights) const {
    return zero_component_oracle(data, weights) + zero_regularizer_oracle(weights);
}

void blackbox::first_oracle(Data* data, double* _pF, bool is_stochastic
        , std::default_random_engine* generator
        , std::uniform_int_distribution<int>* distribution
        , double* weights) const {
    if(weights == NULL) weights = m_weights;
    if(is_stochastic) {
        int rand_samp = (*distribution)(*generator);
        first_oracle(data, _pF, rand_samp, weights);
    }
    else {
        for(size_t i = 0; i < data->size(); i ++) {
            double* _pf = new double[MAX_DIM];
            first_oracle(data, _pf, i, weights);
            for(size_t j = 0; j < MAX_DIM; j ++)
                _pF[j] += _pf[j];
            delete[] _pf;
        }
        for(size_t j = 0; j < MAX_DIM; j ++)
            _pF[j] /= data->size();
    }
}

void blackbox::first_oracle(Data* data, double* _pF, int given_index, double* weights) const {
    if(weights == NULL) weights = m_weights;
    double* _pR = new double[MAX_DIM];
    first_regularizer_oracle(_pR, weights);
    first_component_oracle(data, _pF, given_index, weights);
    for(size_t i = 0; i < MAX_DIM; i ++)
        _pF[i] += _pR[i];
    delete[] _pR;
}

void blackbox::first_component_oracle(Data* data, double* _pF, bool is_stochastic
        , std::default_random_engine* generator
        , std::uniform_int_distribution<int>* distribution
        , double* weights) const {
    if(weights == NULL) weights = m_weights;
    if(is_stochastic) {
        int rand_samp = (*distribution)(*generator);
        first_component_oracle(data, _pF, rand_samp, weights);
    }
    else {
        for(size_t i = 0; i < data->size(); i ++) {
            double* _pf = new double[MAX_DIM];
            first_component_oracle(data, _pf, i, weights);
            for(size_t j = 0; j < MAX_DIM; j ++)
                _pF[j] += _pf[j];
            delete[] _pf;
        }
        for(size_t j = 0; j < MAX_DIM; j ++)
            _pF[j] /= data->size();
    }
}

double blackbox::zero_regularizer_oracle(double* weights) const {
    if(weights == NULL) weights = m_weights;
    return regularizer::zero_oracle(m_regularizer, *m_params, weights);
}

void blackbox::first_regularizer_oracle(double* _pR, double* weights) const {
    if(weights == NULL) weights = m_weights;
    regularizer::first_oracle(m_regularizer, _pR, *m_params, weights);
}

void blackbox::proximal_regularizer(double* _prox, double step_size) const {
    regularizer::proximal_operator(m_regularizer, _prox, *m_params, step_size);
}

void blackbox::set_init_weights(double* init_weights) {
    update_model(init_weights);
}

double* blackbox::get_model() const {
    return m_weights;
}

void blackbox::update_model(double* new_weights) {
    for(size_t i = 0; i < MAX_DIM; i ++)
        m_weights[i] = new_weights[i];
}
