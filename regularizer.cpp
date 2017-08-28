#include <assert.h>
#include <stdio.h>
#include <string.h>
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
double regularizer::proximal_operator(int _regular, double& _prox, double lambda
    , double step_size, size_t times, double additional_constant, bool is_lazy
    , bool is_lazy_weighted, double* lazy_average_weight) {
    double lazy_average = 0.0;
    switch(_regular) {
        case regularizer::L1: {
            for(size_t i = 0; i < times; i ++) {
                _prox += additional_constant;
                if(_prox > step_size * lambda)
                    _prox -= step_size * lambda;
                else if(_prox < -step_size * lambda)
                    _prox += step_size * lambda;
                else
                    _prox = 0;
                double weight = 1.0;
                if(is_lazy_weighted)
                    weight = lazy_average_weight[i];
                if(is_lazy)
                    lazy_average += weight * _prox;
            }
            if(is_lazy)
                return lazy_average;
            else return 0.0;
            break;
        }
        case regularizer::L2: {
            double param = pow(1.0 / (1 + step_size * lambda), times);
            if(is_lazy) {
                if(!is_lazy_weighted) {
                    double param_2 = additional_constant / (lambda * step_size);
                    lazy_average = (_prox - param_2) * (1 - pow(1.0 / (1 + step_size * lambda), times))
                                 / (lambda * step_size) + param_2 * times;
                }
                else {
                    double param_2 = 1.0 / (1 + step_size * lambda);
                    double _tx = _prox;
                    for(size_t i = 0; i < times; i ++) {
                        _tx = param_2 * (_tx + additional_constant);
                        lazy_average += lazy_average_weight[i] * _tx;
                    }
                }
            }
            _prox = _prox * param + additional_constant * (1 - param) / (lambda * step_size);
            if(is_lazy)
                return lazy_average;
            else return 0.0;
            break;
        }
        default:
            return 0.0;
            break;
    }
}
