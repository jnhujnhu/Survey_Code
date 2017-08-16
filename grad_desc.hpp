#ifndef GRADDESC_H
#define GRADDESC_H
#include "blackbox.hpp"
#include "data.hpp"

namespace grad_desc {
    double* GD(Data* data, blackbox* model, size_t iteration_no = 1000
        , bool is_store_weight = false, bool is_debug_mode = false);
    double* SGD(Data* data, blackbox* model, size_t iteration_no = 10000
        , bool is_store_weight = false, bool is_debug_mode = false);
    std::vector<double>* SVRG(Data* data, blackbox* model, size_t outter_iteration_no = 7
        , bool is_store_weight = false, bool is_debug_mode = false);
    std::vector<double>* Katyusha(Data* data, blackbox* model, size_t outter_iteration_no = 7
        , bool is_store_weight = false, bool is_debug_mode = false);
    std::vector<double>* SAG(Data* data, blackbox* model, bool is_store_weight = false
        , bool is_debug_mode = false);
    std::vector<double>* SAGA(Data* data, blackbox* model, bool is_store_weight = false
        , bool is_debug_mode = false);
}

#endif
