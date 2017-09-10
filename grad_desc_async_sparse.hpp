#ifndef GRADDESCASYNCSPARSE_H
#define GRADDESCASYNCSPARSE_H
#include "blackbox.hpp"

namespace grad_desc_async_sparse {
    const int SVRG_LAST_LAST = 1;
    const int SVRG_AVER_AVER = 2;
    const int SVRG_AVER_LAST = 3;
    // For Prox_SVRG / SVRG, Mode 1: last_iter--last_iter, Mode 2: aver_iter--aver_iter, Mode 3: aver_iter--last_iter
    std::vector<double>* Prox_ASVRG(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N, blackbox* model, size_t iteration_no
        , int Mode = 1, double L = 1.0, double step_size = 1.0, bool is_store_weight = false, bool is_debug_mode = false
        , bool is_store_result = false);
    std::vector<double>* A_Katyusha(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N, blackbox* model, size_t iteration_no
        , double L = 1.0, double sigma = 0.0001, double step_size = 1.0, bool is_store_weight = false, bool is_debug_mode = false
        , bool is_store_result = false);
    std::vector<double>* ASVRG(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N, blackbox* model, size_t iteration_no
        , int Mode = 1, double L = 1.0, double step_size = 1.0, bool is_store_weight = false, bool is_debug_mode = false
        , bool is_store_result = false);
    std::vector<double>* ASAGA(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N, blackbox* model, size_t iteration_no
        , double L, double step_size, bool is_store_weight = false, bool is_debug_mode = false
        , bool is_store_result = false);
}

#endif
