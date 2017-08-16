#ifndef UTILS_HPP
#define UTILS_HPP
#include <math.h>
#include <stdio.h>
#include <vector>

extern size_t MAX_DIM;
inline double comp_l2_norm(std::vector<double>* vec) {
    double res = 0.0;
    for(std::vector<double>::iterator iter = (*vec).begin(); iter != (*vec).end()
    ; iter ++) {
        res += (*iter) * (*iter);
    }
    return sqrt(res);
}

inline double comp_l2_norm(double* vec) {
    double res = 0.0;
    for(size_t i = 0; i < MAX_DIM; i ++){
        res += vec[i] * vec[i];
    }
    return sqrt(res);
}

#endif
