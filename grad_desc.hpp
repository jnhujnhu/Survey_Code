#ifndef GRADDESC_H
#define GRADDESC_H
#include "blackbox.hpp"
#include "data.hpp"

namespace grad_desc {
    double* GD(Data* data, blackbox* model, size_t iteration_no = 10000, double L = 1.0, double step_size = 1.0
        , bool is_store_weight = false, bool is_debug_mode = false, bool is_store_result = false);
    double* SGD(Data* data, blackbox* model, size_t iteration_no = 10000, double L = 1.0, double step_size = 1.0
        , bool is_store_weight = false, bool is_debug_mode = false, bool is_store_result = false);
    double* KGD(Data* data, blackbox* model, size_t iteration_no = 10000, double L = 1.0, double step_size = 1.0
        , bool is_store_weight = false, bool is_debug_mode = false, bool is_store_result = false);
    std::vector<double>* SVRG(Data* data, blackbox* model, size_t& iteration_no, double L = 1.0, double step_size = 1.0
        , bool is_store_weight = false, bool is_debug_mode = false, bool is_store_result = false);
    std::vector<double>* Katyusha(Data* data, blackbox* model, size_t& iteration_no, double L = 1.0, double step_size = 1.0
        , bool is_store_weight = false, bool is_debug_mode = false, bool is_store_result = false);
    std::vector<double>* SAG(Data* data, blackbox* model, double L, double step_size, bool is_store_weight = false
        , bool is_debug_mode = false, bool is_store_result = false);
    std::vector<double>* SAGA(Data* data, blackbox* model, double L, double step_size, bool is_store_weight = false
        , bool is_debug_mode = false, bool is_store_result = false);
}

#endif
