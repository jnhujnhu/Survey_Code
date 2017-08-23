#ifndef UTILS_HPP
#define UTILS_HPP
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
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

inline double comp_l1_norm(double* vec) {
    double res = 0.0;
    for(size_t i = 0; i < MAX_DIM; i ++){
        res += std::abs(vec[i]);
    }
    return res;
}

inline void copy_vec(double* vec_to, double* vec_from) {
    for(size_t i = 0; i < MAX_DIM; i ++)
        vec_to[i] = vec_from[i];
}

inline constexpr unsigned int _hash(const char* str, int h = 0) {
    return !str[h] ? 5381 : (_hash(str, h+1) * 33) ^ str[h];
}

#endif
