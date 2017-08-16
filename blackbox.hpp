#ifndef BLACK_BOX_H
#define BLACK_BOX_H
#include "data.hpp"
#include <random>

extern size_t MAX_DIM;
class blackbox {
public:
    virtual ~blackbox();
    virtual int classify(double* sample) const = 0;
    virtual double zero_oracle(Data* data, double* weights = NULL) const;
    virtual double zero_component_oracle(Data* data, double* weights = NULL) const = 0;
    virtual double zero_regularizer_oracle(double* weights = NULL) const = 0;
    virtual void first_oracle(Data* data, double* _pF, bool is_stochastic = false
            , std::default_random_engine* generator = NULL
            , std::uniform_int_distribution<int>* distribution = NULL
            , double* weights = NULL) const;
    virtual void first_oracle(Data* data, double* _pF, int given_index, double* weights = NULL) const;
    virtual void first_component_oracle(Data* data, double* _pF, int given_index, double* weights = NULL) const = 0;
    virtual void first_regularizer_oracle(double* _pR, double* weights = NULL) const = 0;
    virtual void proximal_regularizer(double* _prox, double step_size) const = 0;
    void set_init_weights(double* init_weights);
    double* get_model() const;
    void update_model(double* new_weights);
protected:
    double* m_weights;
    double* m_params;
};


#endif
