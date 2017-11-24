#ifndef GRADDESCMiG_H
#define GRADDESCMiG_H
#include "blackbox.hpp"

namespace grad_desc_MiG {
    const int SVRG_LAST_LAST = 1;
    const int SVRG_AVER_AVER = 2;
    const int SVRG_AVER_LAST = 3;

    std::vector<double>* Ladder_SVRG(double* X, double* Y, size_t N, blackbox* model
        , size_t iteration_no, int Mode, double L, double sigma, double step_size
        , bool is_store_result);
    std::vector<double>* Ladder_SVRG_sparse(double* X, double* Y, size_t* Jc, size_t* Ir
        , size_t N, blackbox* model, size_t iteration_no, int Mode, double L, double sigma, double step_size
        , bool is_store_result);
}

#endif
