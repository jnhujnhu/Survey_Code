#include "grad_desc.hpp"
#include "utils.hpp"
#include <string.h>

extern size_t MAX_DIM;

// w = A*w + Const
double lazy_update_SVRG(double& w, double Const, double A, size_t times) {
    double pow_A = pow(A, times);
    double T1 = A * (1 - pow_A) / (1 - A);
    double T2 = Const / (1 - A);
    double lazy_average = T1 * w + T2 + T1 * T2;
    w = pow_A * w + Const * (1 - pow_A) / (1 - A);
    return lazy_average;
}

// Only Applicable for L2 regularizer
std::vector<double>* grad_desc::SVRG(Data* data, blackbox* model, size_t& iteration_no, int Mode
    , double L, double step_size, bool is_store_weight, bool is_debug_mode, bool is_store_result) {
        // Random Generator
        std::random_device rd;
        std::default_random_engine generator(rd());
        std::uniform_int_distribution<int> distribution(0, data->size() - 1);
        std::vector<double>* stored_weights = new std::vector<double>;
        std::vector<double>* stored_F = new std::vector<double>;
        double* inner_weights = new double[MAX_DIM];
        double* full_grad = new double[MAX_DIM];
        size_t N = data->size();
        double lambda = model->get_param(0);
        //FIXME: Epoch Size(SVRG / SVRG++)
        double m0 = (double) N * 2.0;
        size_t total_iterations = 0;
        copy_vec(inner_weights, model->get_model());
        // OUTTER_LOOP
        for(size_t i = 0 ; i < iteration_no; i ++) {
            double* full_grad_core = new double[N];
            double* outter_weights = (model->get_model());
            // Average Iterates
            double* aver_weights = new double[MAX_DIM];
            // lazy update extra params.
            int* last_seen = new int[MAX_DIM];
            //FIXME: SVRG / SVRG++
            double inner_m = m0;//pow(2, i + 1) * m0;
            memset(aver_weights, 0, MAX_DIM * sizeof(double));
            memset(full_grad, 0, MAX_DIM * sizeof(double));
            for(size_t i = 0; i < MAX_DIM; i ++) last_seen[i] = -1;
            // Full Gradient
            for(size_t j = 0; j < N; j ++) {
                full_grad_core[j] = model->first_component_oracle_core(data, j);
                for(Data::iterator iter = (*data)(j); iter.hasNext();) {
                    size_t index = iter.getIndex();
                    full_grad[index] += (iter.next() * full_grad_core[j]
                                     + lambda * outter_weights[index]) / (double) N;
                }
            }
            switch(Mode) {
                case SVRG_LAST_LAST:
                case SVRG_AVER_LAST:
                    break;
                case SVRG_AVER_AVER:
                    copy_vec(inner_weights, model->get_model());
                    break;
                default:
                    throw std::string("Unrecognized Mode.");
                    break;
            }
            // INNER_LOOP
            for(size_t j = 0; j < inner_m ; j ++) {
                int rand_samp = distribution(generator);
                double inner_core = model->first_component_oracle_core(data, rand_samp, inner_weights);
                for(Data::iterator iter = (*data)(rand_samp); iter.hasNext();) {
                    size_t index = iter.getIndex();
                    double val = iter.next();
                    // lazy update
                    if((int)j > last_seen[index] + 1) {
                        aver_weights[index] += lazy_update_SVRG(inner_weights[index]
                            , step_size * (lambda * outter_weights[index] - full_grad[index])
                            , 1 - step_size * lambda, j - (last_seen[index] + 1)) / inner_m;
                    }
                    double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val
                             + (inner_weights[index] - outter_weights[index]) * lambda + full_grad[index];
                    inner_weights[index] -= step_size * vr_sub_grad;
                    aver_weights[index] += inner_weights[index] / inner_m;
                    last_seen[index] = j;
                }
                total_iterations ++;
                // For Drawing
                if(is_store_weight) {
                    for(size_t k = 0; k < MAX_DIM; k ++)
                        stored_weights->push_back(inner_weights[k]);
                }
                // For Matlab
                // if(is_store_result) {
                //     if(!(total_iterations % N)) {
                //         stored_F->push_back(model->zero_oracle(data, inner_weights));
                //     }
                // }
                if(is_debug_mode) {
                    double log_F = log(model->zero_oracle(data, inner_weights));
                    printf("SVRG: Outter Iteration: %zd -> Inner Iteration %zd, log_F for inner_weights: %lf.\n", i, j, log_F);
                }
            }
            // lazy update aggragate
            for(size_t j = 0; j < MAX_DIM; j ++) {
                if(inner_m > last_seen[j] + 1) {
                    aver_weights[j] += lazy_update_SVRG(inner_weights[j]
                        , step_size * (lambda * outter_weights[j] - full_grad[j])
                        , 1 - step_size * lambda, inner_m - (last_seen[j] + 1)) / inner_m;
                }
            }
            switch(Mode) {
                case SVRG_LAST_LAST:
                    model->update_model(inner_weights);
                    break;
                case SVRG_AVER_AVER:
                case SVRG_AVER_LAST:
                    model->update_model(aver_weights);
                    break;
                default:
                    throw std::string("Unrecognized Mode.");
                    break;
            }
            // FIXME: Test
            if(is_store_result) {
                if(!(total_iterations % N)) {
                    stored_F->push_back(model->zero_oracle(data));
                }
            }
            delete[] last_seen;
            delete[] aver_weights;
            delete[] full_grad_core;
            if(is_debug_mode) {
                double log_F = log(model->zero_oracle(data));
                printf("SVRG: Outter Iteration %zd, log_F: %lf.\n", i, log_F);
            }
        }
        delete[] full_grad;
        delete[] inner_weights;
        //Final Output
        double log_F = log(model->zero_oracle(data));
        iteration_no = total_iterations;
        printf("SVRG: Total Iteration No.: %zd, logF = %lf.\n", total_iterations, log_F);
        if(is_store_weight)
            return stored_weights;
        if(is_store_result)
            return stored_F;
        return NULL;
}

std::vector<double>* grad_desc::SAG(Data* data, blackbox* model, double L
    , double step_size, bool is_store_weight, bool is_debug_mode, bool is_store_result) {
    //TODO:
    return NULL;
}

std::vector<double>* grad_desc::SAGA(Data* data, blackbox* model, double L
    , double step_size, bool is_store_weight, bool is_debug_mode, bool is_store_result) {
    //TODO:
    return NULL;
}
