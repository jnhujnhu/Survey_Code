#ifndef GRADDESCASYNCSPARSE_H
#define GRADDESCASYNCSPARSE_H
#include "blackbox.hpp"
#include <iostream>
#include <atomic>

namespace grad_desc_async_sparse {
    const int SVRG_LAST_LAST = 1;
    const int SVRG_AVER_AVER = 2;
    const int SVRG_AVER_LAST = 3;

    void Partial_Gradient(double* full_grad_core, size_t thread_no, double* X, double* Y, size_t* Jc, size_t* Ir
        , std::atomic<double>* full_grad, size_t N, blackbox* model, size_t _thread, double* _weights = NULL
        , std::atomic<double>* reweight_diag = NULL);
    double* Comp_Full_Grad_Parallel(double* full_grad_core, size_t thread_no, double* X, double* Y, size_t* Jc, size_t* Ir
        , size_t N, blackbox* model, double* _weights = NULL, std::atomic<double>* reweight_diag = NULL);

    // For Prox_SVRG / SVRG, Mode 1: last_iter--last_iter, Mode 2: aver_iter--aver_iter, Mode 3: aver_iter--last_iter
    std::vector<double>* Prox_ASVRG(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N, blackbox* model, size_t iteration_no
        , size_t thread_no, int Mode = 1, double L = 1.0, double step_size = 1.0, bool is_store_result = false);
    std::vector<double>* Prox_ASVRG_Single(double* X, double* Y, size_t* Jc, size_t* Ir
        , size_t N, blackbox* model, size_t iteration_no, int Mode, double L, double step_size
        , bool is_store_result);
    std::vector<double>* Prox_ASVRG_Async(double* X, double* Y, size_t* Jc, size_t* Ir
        , size_t N, blackbox* model, size_t iteration_no, size_t thread_no, int Mode, double L, double step_size
        , bool is_store_result);
    void Prox_ASVRG_Async_Inner_Loop(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N
        , std::atomic<double>* x, std::atomic<double>* aver_x, blackbox* model, size_t m
        , size_t inner_iters, double step_size, std::atomic<double>* reweight_diag
        , double* full_grad_core, double* full_grad);

    std::vector<double>* ASAGA_Async(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N, blackbox* model, size_t iteration_no
        , size_t thread_no, double L, double step_size, bool is_store_result = false);
    std::vector<double>* ASAGA_Single(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N, blackbox* model, size_t iteration_no
        , double L, double step_size, bool is_store_result = false);
    std::vector<double>* ASAGA(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N, blackbox* model, size_t iteration_no
        , size_t thread_no, double L, double step_size, bool is_store_result = false);
    void ASAGA_Async_Loop(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N
        , std::atomic<double>* x, blackbox* model, size_t inner_iters, double step_size
        , double* reweight_diag, std::atomic<double>* grad_core_table, std::atomic<double>* aver_grad
        , std::vector<double>* stored_F, bool is_store_result);

    // FIXME: Diverge if without lock with more than 3 threads.
    std::vector<double>* A_Katyusha(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N, blackbox* model, size_t iteration_no
        , size_t thread_no, double L = 1.0, double sigma = 0.0001, double step_size = 1.0, bool is_store_result = false);
    std::vector<double>* A_Katyusha_Single(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N, blackbox* model, size_t iteration_no
        , double L = 1.0, double sigma = 0.0001, double step_size = 1.0, bool is_store_result = false);
    std::vector<double>* A_Katyusha_Async(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N, blackbox* model, size_t iteration_no
        , size_t thread_no, double L = 1.0, double sigma = 0.0001, double step_size = 1.0, bool is_store_result = false);
    void A_Katyusha_Async_Inner_Loop(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N, std::atomic<double>* x, std::atomic<double>* y
        , std::atomic<double>* z, std::atomic<double>* aver_y, blackbox* model, size_t m, size_t inner_iters
        , double step_size, double tau_1, double tau_2, double alpha, double compos_factor
        , double compos_base, std::atomic<double>* reweight_diag, double* full_grad_core
        , double* full_grad, double* compos_pow, double* outter_x);

    // std::vector<double>* ASVRG(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N, blackbox* model, size_t iteration_no
    //     , size_t thread_no, int Mode = 1, double L = 1.0, double step_size = 1.0, bool is_store_result = false);
}

#endif
