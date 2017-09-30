#ifndef GRADDESCSDSPARSE_H
#define GRADDESCSDSPARSE_H
#include "blackbox.hpp"

namespace grad_desc_sd_sparse {
    // Only Implemented Ridge Regression
    std::vector<double>* SVRG_SD(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N, blackbox* model, size_t iteration_no
        , double L, double step_size, bool is_store_weight = false, bool is_debug_mode = false, bool is_store_result = false);
    std::vector<double>* SAGA_SD(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N, blackbox* model, size_t iteration_no
        , double L, double step_size, double r, double* SV, bool is_store_weight = false, bool is_debug_mode = false
        , bool is_store_result = false);
}

#endif
