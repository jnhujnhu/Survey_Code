#ifndef BLACK_BOX_H
#define BLACK_BOX_H
#include "data.hpp"
#include <random>
#include <stdio.h>

extern size_t MAX_DIM;
class blackbox {
public:
    virtual ~blackbox() {
        delete[] m_weights;
        delete[] m_params;
    }
    virtual int classify(double* sample) const = 0;
    virtual double zero_oracle(Data* data, double* weights = NULL) const = 0;
    virtual void first_oracle(Data* data, double* _pF, bool is_stochastic = false
            , std::default_random_engine* generator = NULL
            , std::uniform_int_distribution<int>* distribution = NULL
            , double* weights = NULL) const = 0;
    virtual void first_oracle(Data* data, double* _pF, int given_index, double* weights = NULL) const = 0;
    double* get_model() const {
        return m_weights;
    }
    void update_model(double* new_weights) {
        for(size_t i = 0; i < MAX_DIM; i ++)
            m_weights[i] = new_weights[i];
    }
protected:
    double* m_weights;
    double* m_params;
};


#endif
