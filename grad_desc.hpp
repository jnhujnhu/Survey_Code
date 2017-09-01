#ifndef GRADDESC_H
#define GRADDESC_H
#include "blackbox.hpp"
#include "data.hpp"

namespace grad_desc {
    const int SVRG_LAST_LAST = 1;
    const int SVRG_AVER_AVER = 2;
    const int SVRG_AVER_LAST = 3;
    // in grad_desc.cpp
    double* GD(Data* data, blackbox* model, size_t iteration_no = 10000, double L = 1.0, double step_size = 1.0
        , bool is_store_weight = false, bool is_debug_mode = false, bool is_store_result = false);
    double* SGD(Data* data, blackbox* model, size_t iteration_no = 10000, double L = 1.0, double step_size = 1.0
        , bool is_store_weight = false, bool is_debug_mode = false, bool is_store_result = false);
    // For Prox_SVRG, Mode 1: last_iter--last_iter, Mode 2: aver_iter--aver_iter, Mode 3: aver_iter--last_iter
    std::vector<double>* Prox_SVRG(Data* data, blackbox* model, size_t& iteration_no, int Mode = 1, double L = 1.0, double step_size = 1.0
        , bool is_store_weight = false, bool is_debug_mode = false, bool is_store_result = false);
    std::vector<double>* Katyusha(Data* data, blackbox* model, size_t& iteration_no, double L = 1.0, double sigma = 0.0001
        , double step_size = 1.0, bool is_store_weight = false, bool is_debug_mode = false, bool is_store_result = false);
    // in grad_desc2.cpp
    // For SVRG, Mode 1: last_iter--last_iter, Mode 2: aver_iter--aver_iter, Mode 3: aver_iter--last_iter
    std::vector<double>* SVRG(Data* data, blackbox* model, size_t& iteration_no, int Mode = 1, double L = 1.0, double step_size = 1.0
        , bool is_store_weight = false, bool is_debug_mode = false, bool is_store_result = false);
    std::vector<double>* SAG(Data* data, blackbox* model, double L, double step_size, bool is_store_weight = false
        , bool is_debug_mode = false, bool is_store_result = false);
    std::vector<double>* SAGA(Data* data, blackbox* model, double L, double step_size, bool is_store_weight = false
        , bool is_debug_mode = false, bool is_store_result = false);
}

#endif
