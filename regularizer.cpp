#include <assert.h>
#include <stdio.h>
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
void regularizer::proximal_operator(int _regular, double* _prox, double lambda, double step_size) {
    switch(_regular) {
        case regularizer::L1: {
            for(size_t i = 0; i < MAX_DIM; i ++) {
                if(_prox[i] > step_size * lambda)
                    _prox[i] -= step_size * lambda;
                else if(_prox[i] < -step_size * lambda)
                    _prox[i] += step_size * lambda;
                else
                    _prox[i] = 0;
            }
            break;
        }
        case regularizer::L2: {
            for(size_t i = 0; i < MAX_DIM; i ++) {
                _prox[i] /= (1 + step_size * lambda);
            }
            break;
        }
        default:
            break;
    }
}
