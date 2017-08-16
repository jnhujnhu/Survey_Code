#include "grad_desc.hpp"
#include <random>
#include <math.h>

extern size_t MAX_DIM;
double* grad_desc::GD(Data* data, blackbox* model, size_t iteration_no
    , bool is_store_weight, bool is_debug_mode) {
    double* stored_weights = NULL;
    if(is_store_weight)
        stored_weights = new double[iteration_no * MAX_DIM];
    double* full_grad = new double[MAX_DIM];
    double* new_weights = new double[MAX_DIM];
    for(size_t i = 0; i < iteration_no; i ++) {
        model->first_oracle(data, full_grad);
        //FIXME:
        double step_size = 1.0 / 1.0;

        for(size_t j = 0; j < MAX_DIM; j ++) {
            new_weights[j] = (model->get_model())[j] - step_size * full_grad[j];
        }
        model->update_model(new_weights);
        if(is_debug_mode) {
            double log_F = log(model->zero_oracle(data));
            printf("GD: Iteration %zd, log_F: %lf.\n", i, log_F);
        }
        if(is_store_weight) {
            for(size_t j = 0; j < MAX_DIM; j ++) {
                stored_weights[i * MAX_DIM + j] = new_weights[j];
            }
        }
    }
    delete[] full_grad;
    delete[] new_weights;
    return stored_weights;
}

double* grad_desc::SGD(Data* data, blackbox* model, size_t iteration_no
    , bool is_store_weight, bool is_debug_mode) {
    //Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, data->size() - 1);

    double* stored_weights = NULL;
    if(is_store_weight)
        stored_weights = new double[iteration_no * MAX_DIM];

    double* sub_grad = new double[MAX_DIM];
    double* new_weights = new double[MAX_DIM];
    for(size_t i = 0; i < iteration_no; i ++) {
        model->first_oracle(data, sub_grad, true, &generator, &distribution);
        //FIXME: Modify step_size.
        double step_size = 1.0 / (1.0 + 0.01 * i);

        for(size_t j = 0; j < MAX_DIM; j ++) {
            new_weights[j] = (model->get_model())[j] - step_size * sub_grad[j];
        }
        model->update_model(new_weights);
        if(is_debug_mode) {
            double log_F = log(model->zero_oracle(data));
            printf("SGD: Iteration %zd, log_F: %lf.\n", i, log_F);
        }
        if(is_store_weight) {
            for(size_t j = 0; j < MAX_DIM; j ++) {
                stored_weights[i * MAX_DIM + j] = new_weights[j];
            }
        }
    }
    delete[] sub_grad;
    delete[] new_weights;
    return stored_weights;
}

std::vector<double>* grad_desc::SVRG(Data* data, blackbox* model, size_t outter_iteration_no
    , bool is_store_weight, bool is_debug_mode) {
    //Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, data->size() - 1);

    std::vector<double>* stored_weights = new std::vector<double>(0);

    double* inner_weights = new double[MAX_DIM];
    double* vr_sub_grad = new double[MAX_DIM];
    double* full_grad = new double[MAX_DIM];
    double* inner_sub_grad = new double[MAX_DIM];
    double* outter_sub_grad = new double[MAX_DIM];
    //FIXME:
    double m0 = (double) data->size() / 4.0;
    size_t total_iterations = 0;
    //OUTTER_LOOP
    for(size_t i = 0 ; i < outter_iteration_no; i ++) {
        model->first_oracle(data, full_grad);
        //FIXME:
        double inner_m = m0;//pow(2, i + 1) * m0;
        double* new_weights = new double[MAX_DIM];
        for(size_t k = 0; k < MAX_DIM; k ++) {
            inner_weights[k] = (model->get_model())[k];
        }
        //INNER_LOOP
        for(size_t j = 0; j < inner_m ; j ++) {
            int rand_samp = distribution(generator);
            model->first_oracle(data, inner_sub_grad, rand_samp, inner_weights);
            model->first_oracle(data, outter_sub_grad, rand_samp);
            for(size_t k = 0; k < MAX_DIM; k ++) {
                vr_sub_grad[k] = inner_sub_grad[k] - outter_sub_grad[k] + full_grad[k];
            }
            double step_size = 1.0 / 1.0;
            for(size_t k = 0; k < MAX_DIM; k ++) {
                inner_weights[k] -= step_size * vr_sub_grad[k];
                new_weights[k] += inner_weights[k] / inner_m;
            }
            if(is_store_weight) {
                size_t prev_size = stored_weights->size();
                stored_weights->resize(prev_size + MAX_DIM);
                for(size_t k = 0; k < MAX_DIM; k ++)
                    (*stored_weights)[prev_size + k] = inner_weights[k];
            }
            if(is_debug_mode) {
                double log_F = log(model->zero_oracle(data, inner_weights));
                printf("SVRG: Outter Iteration: %zd -> Inner Iteration %zd, log_F for inner_weights: %lf.\n", i, j, log_F);
            }
            total_iterations ++;
        }
        //FIXME:
        model->update_model(inner_weights);
        delete[] new_weights;
        if(is_debug_mode) {
            double log_F = log(model->zero_oracle(data));
            printf("SVRG: Outter Iteration %zd, log_F: %lf.\n", i, log_F);
        }
    }
    delete[] inner_sub_grad;
    delete[] outter_sub_grad;
    delete[] vr_sub_grad;
    delete[] full_grad;
    delete[] inner_weights;
    if(is_debug_mode)
        printf("Total Iteration No.: %zd\n", total_iterations);
    if(is_store_weight)
        return stored_weights;
    return NULL;
}

std::vector<double>* grad_desc::SAG(Data* data, blackbox* model, bool is_store_weight
    , bool is_debug_mode) {
    //TODO
    return NULL;
}

std::vector<double>* grad_desc::SAGA(Data* data, blackbox* model, bool is_store_weight
    , bool is_debug_mode) {
    //TODO
    return NULL;
}
