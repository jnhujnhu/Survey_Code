#ifndef UTILS_HPP
#define UTILS_HPP
#include <math.h>

inline double comp_l2_norm(std::vector<double>* vec) {
    double res = 0.0;
    for(std::vector<double>::iterator iter = (*vec).begin(); iter != (*vec).end()
    ; iter ++) {
        res += (*iter) * (*iter);
    }
    return sqrt(res);
}

#endif
