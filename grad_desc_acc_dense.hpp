#ifndef GRADDESCACCDENSE_H
#define GRADDESCACCDENSE_H
#include "blackbox.hpp"

namespace grad_desc_acc_dense {
    const int SVRG_LAST_LAST = 1;
    const int SVRG_AVER_AVER = 2;
    const int SVRG_AVER_LAST = 3;

    const int SVRG_LS_FULL = 4;
    const int SVRG_LS_STOC = 5;

    const int SVRG_LS_CHGF = 6;
    const int SVRG_LS_OUTF = 7;

    const int SVRG_LS_SVD = 8;
    const int SVRG_LS_A = 9;

    std::vector<double>* Acc_Prox_SVRG1(double* X, double* Y, size_t N, blackbox* model, size_t iteration_no
        , double L, double sigma, double step_size, bool is_store_result = false);
    // Only for Ridge Regression
    std::vector<double>* SVRG_LS(double* X, double* Y, size_t N, blackbox* model, size_t iteration_no
        , size_t interval, int Mode, int LSF_Mode, int LSC_Mode, int LSM_Mode
        , double L, double step_size, double r, double* SV, bool is_store_result);
}

#endif
