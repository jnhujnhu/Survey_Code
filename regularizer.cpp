#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "regularizer.hpp"
#include "utils.hpp"

extern size_t MAX_DIM;

double regularizer::zero_oracle(int _regular, double* lambda, double* weight) {
    assert (weight != NULL);
    switch(_regular) {
        case regularizer::L1: {
            return lambda[1] * comp_l1_norm(weight);
            break;
        }
        case regularizer::L2: {
            double l2_norm = comp_l2_norm(weight);
            return lambda[0] * 0.5 * l2_norm * l2_norm;
            break;
        }
        case regularizer::ELASTIC_NET: {
            double l2_norm = comp_l2_norm(weight);
            double l2_part = lambda[0] * 0.5 * l2_norm * l2_norm;
            double l1_part = lambda[1] * comp_l1_norm(weight);
            return l1_part + l2_part;
            break;
        }
        default:
            return 0;
    }
}
void regularizer::first_oracle(int _regular, double* _pR, double* lambda, double* weight) {
    assert (weight != NULL);
    memset(_pR, 0, MAX_DIM * sizeof(double));
    switch(_regular) {
        case regularizer::L1: {
            //Not Available
            break;
        }
        case regularizer::L2: {
            for(size_t i = 0; i < MAX_DIM; i ++) {
                _pR[i] = lambda[0] * weight[i];
            }
            break;
        }
        case regularizer::ELASTIC_NET: {
            //Not Available
            break;
        }
        default:
            break;
    }
}

double regularizer::L1_proximal_loop(double& _prox, double param, size_t times, double additional_constant,
        bool is_averaged) {
    double lazy_average = 0.0;
    for(size_t i = 0; i < times; i ++) {
        _prox += additional_constant;
        if(_prox > param)
            _prox -= param;
        else if(_prox < -param)
            _prox += param;
        else
            _prox = 0;
        if(is_averaged)
            lazy_average += _prox;
    }
    return lazy_average;
}

double regularizer::proximal_operator(int _regular, double& _prox, double step_size
    , double* lambda) {
    switch(_regular) {
        case regularizer::L1: {
            double param = step_size * lambda[1];
            if(_prox > param)
                _prox -= param;
            else if(_prox < -param)
                _prox += param;
            else
                _prox = 0;
            return _prox;
        }
        case regularizer::L2: {
            _prox = _prox / (1 + step_size * lambda[0]);
            return _prox;
        }
        case regularizer::ELASTIC_NET: {
            double param_1 = step_size * lambda[1];
            double param_2 = 1.0 / (1.0 + step_size * lambda[0]);
            if(_prox > param_1)
                _prox = param_2 * (_prox - param_1);
            else if(_prox < - param_1)
                _prox = param_2 * (_prox + param_1);
            else
                _prox = 0;
            return _prox;
            break;
        }
        default:
            return 0.0;
            break;
    }
}

// Lazy(Lagged) Update
double regularizer::proximal_operator(int _regular, double& _prox, double step_size
    , double* lambda, size_t times, bool is_averaged, double additional_constant) {
    double lazy_average = 0.0;
    switch(_regular) {
        case regularizer::L1: {
            double param = step_size * lambda[1];
            if(_prox > param - additional_constant){
                if(additional_constant > param) {
                    if(is_averaged)
                        lazy_average = times * _prox - (param - additional_constant)
                                     * (1 + times) * times / 2.0;
                    _prox -= times * (param - additional_constant);
                }
                else {
                    int _ti = floor((-_prox - additional_constant + param) / (additional_constant - param));
                    if((int)times <= _ti) {
                        if(is_averaged)
                            lazy_average = times * _prox - (param - additional_constant)
                                         * (1 + times) * times / 2.0;
                        _prox -= times * (param - additional_constant);
                    }
                    else lazy_average = L1_proximal_loop(_prox, param, times, additional_constant, is_averaged);
                }
            }
            else if(_prox < -param - additional_constant){
                if(additional_constant < -param) {
                    if(is_averaged)
                        lazy_average = times * _prox + (param + additional_constant)
                                     * (1 + times) * times / 2.0;
                    _prox += times * (param + additional_constant);
                }
                else {
                    int _ti = floor((-_prox - additional_constant - param) / (additional_constant + param));
                    if((int)times <= _ti) {
                        if(is_averaged)
                            lazy_average = times * _prox + (param + additional_constant)
                                         * (1 + times) * times / 2.0;
                        _prox += times * (param + additional_constant);
                    }
                    else lazy_average = L1_proximal_loop(_prox, param, times, additional_constant, is_averaged);
                }
            }
            else {
                if(param >= additional_constant && param >= -additional_constant)
                    _prox = 0;
                else lazy_average = L1_proximal_loop(_prox, param, times, additional_constant, is_averaged);
            }
            return lazy_average;
            break;
        }
        case regularizer::L2: {
            if(times == 1) {
                _prox = (_prox + additional_constant) / (1 + step_size * lambda[0]);
                return _prox;
            }
            double param_1 = step_size * lambda[0];
            double param_2 = pow((double) 1.0 / (1 + param_1), (double) times);
            double param_3 = additional_constant / param_1;
            if(is_averaged)
                lazy_average = (_prox - param_3) * (1 - param_2) / param_1 + param_3 * times;
            _prox = _prox * param_2 + param_3 * (1 - param_2);
            return lazy_average;
            break;
        }
        case regularizer::ELASTIC_NET: {
            // Naive Solution
            double param_1 = step_size * lambda[1];
            double param_2 = 1.0 / (1.0 + step_size * lambda[0]);
            for(size_t i = 0; i < times; i ++) {
                _prox += additional_constant;
                if(_prox > param_1)
                    _prox = param_2 * (_prox - param_1);
                else if(_prox < - param_1)
                    _prox = param_2 * (_prox + param_1);
                else
                    _prox = 0;
                if(is_averaged)
                    lazy_average += _prox;
            }
            return lazy_average;
            break;
        }
        default:
            return 0.0;
            break;
    }
}
