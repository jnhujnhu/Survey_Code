#ifndef GRADDESCACCDENSE_H
#define GRADDESCACCDENSE_H
#include "blackbox.hpp"

namespace grad_desc_acc_dense {
    std::vector<double>* Acc_Prox_SVRG1(double* X, double* Y, size_t N, blackbox* model, size_t iteration_no
        , double L, double sigma, double step_size, bool is_store_result = false);
}

#endif
