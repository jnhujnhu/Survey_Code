#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "regularizer.hpp"
#include "utils.hpp"

extern size_t MAX_DIM;

double regularizer::zero_oracle(int _regular, double lambda, double* weight) {
    assert (weight != NULL);
    switch(_regular) {
        case regularizer::L1: {
            return lambda * comp_l1_norm(weight);
            break;
        }
        case regularizer::L2: {
            double l2_norm = comp_l2_norm(weight);
            return lambda * 0.5 * l2_norm * l2_norm;
            break;
        }
        default:
            return 0;
    }
}
void regularizer::first_oracle(int _regular, double* _pR, double lambda, double* weight) {
    assert (weight != NULL);
    memset(_pR, 0, MAX_DIM * sizeof(double));
    switch(_regular) {
        case regularizer::L1: {
            //Not Available
            break;
        }
        case regularizer::L2: {
            for(size_t i = 0; i < MAX_DIM; i ++) {
                _pR[i] = lambda * weight[i];
            }
            break;
        }
        default:
            break;
    }
}

double L1_proximal_loop(double& _prox, double param, size_t times, double additional_constant) {
    double lazy_average = 0.0;
    for(size_t i = 0; i < times; i ++) {
        _prox += additional_constant;
        if(_prox > param)
            _prox -= param;
        else if(_prox < -param)
            _prox += param;
        else
            _prox = 0;
        lazy_average += _prox;
    }
    return lazy_average;
}

double regularizer::proximal_operator(int _regular, double& _prox, double lambda
    , double step_size, size_t times, double additional_constant, bool is_lazy_weighted
    , double* lazy_average_weight) {
    double lazy_average = 0.0;
    switch(_regular) {
        //FIXME: Ignored Lazy Weights to Speed Up.
        case regularizer::L1: {
            double param = step_size * lambda;
            if(_prox > param - additional_constant){
                if(additional_constant > param) {
                    lazy_average = times * _prox - (param - additional_constant)
                                 * (1 + times) * times / 2.0;
                    _prox -= times * (param - additional_constant);
                }
                else {
                    int _ti = floor((-_prox - additional_constant + param) / (additional_constant - param));
                    if((int)times <= _ti) {
                        lazy_average = times * _prox - (param - additional_constant)
                                     * (1 + times) * times / 2.0;
                        _prox -= times * (param - additional_constant);
                    }
                    else lazy_average = L1_proximal_loop(_prox, param, times, additional_constant);
                }
            }
            else if(_prox < -param - additional_constant){
                if(additional_constant < -param) {
                    lazy_average = times * _prox + (param + additional_constant)
                                 * (1 + times) * times / 2.0;
                    _prox += times * (param + additional_constant);
                }
                else {
                    int _ti = floor((-_prox - additional_constant - param) / (additional_constant + param));
                    if((int)times <= _ti) {
                        lazy_average = times * _prox + (param + additional_constant)
                                     * (1 + times) * times / 2.0;
                        _prox += times * (param + additional_constant);
                    }
                    else lazy_average = L1_proximal_loop(_prox, param, times, additional_constant);
                }
            }
            else {
                if(param >= additional_constant && param >= -additional_constant)
                    _prox = 0;
                else lazy_average = L1_proximal_loop(_prox, param, times, additional_constant);
            }
            return lazy_average;
            break;
        }
        //FIXME: Ignored Lazy Weights to Speed Up.
        case regularizer::L2: {
            double param = pow((double) 1.0 / (1 + step_size * lambda), (double) times);
            double param_2 = additional_constant / (lambda * step_size);
            lazy_average = (_prox - param_2) * (1 - param) / (lambda * step_size)
                         + param_2 * times;
            _prox = _prox * param + param_2 * (1 - param);
            return lazy_average;
            break;
        }
        default:
            return 0.0;
            break;
    }
}
