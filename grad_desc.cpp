#include "grad_desc.hpp"
#include "utils.hpp"
#include <random>
#include <math.h>

extern size_t MAX_DIM;
double* grad_desc::GD(Data* data, blackbox* model, size_t iteration_no
    , bool is_store_weight, bool is_debug_mode) {
    double* stored_weights = NULL;
    if(is_store_weight)
        stored_weights = new double[iteration_no * MAX_DIM];
    double* new_weights = new double[MAX_DIM];
    for(size_t i = 0; i < iteration_no; i ++) {
        double* full_grad = new double[MAX_DIM];
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
        else
            printf("GD: Iteration %zd.\n", i);
        if(is_store_weight) {
            for(size_t j = 0; j < MAX_DIM; j ++) {
                stored_weights[i * MAX_DIM + j] = new_weights[j];
            }
        }
        delete[] full_grad;
    }
    //Final Output
    double log_F = log(model->zero_oracle(data));
    printf("GD: Iteration %zd, log_F: %lf.\n", iteration_no, log_F);
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
        double step_size = 1.0 / 1.0;

        for(size_t j = 0; j < MAX_DIM; j ++) {
            new_weights[j] = (model->get_model())[j] - step_size * sub_grad[j];
        }
        printf("Norm: %lf ", comp_l2_norm(sub_grad));
        model->update_model(new_weights);
        if(is_debug_mode) {
            double log_F = log(model->zero_oracle(data));
            printf("SGD: Iteration %zd, log_F: %lf.\n", i, log_F);
        }
        else
            printf("SGD: Iteration %zd.\n", i);
        if(is_store_weight) {
            for(size_t j = 0; j < MAX_DIM; j ++) {
                stored_weights[i * MAX_DIM + j] = new_weights[j];
            }
        }
    }
    //Final Output
    double log_F = log(model->zero_oracle(data));
    printf("SGD: Iteration %zd, log_F: %lf.\n", iteration_no, log_F);
    delete[] sub_grad;
    delete[] new_weights;
    return stored_weights;
}

double* grad_desc::KGD(Data* data, blackbox* model, size_t iteration_no
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

    size_t m = data->size() / 2.0, _m = 0;
    double* k_grad = new double[MAX_DIM];
    double _param1 = 2.0, _param2 = 0.1;
    double variety = 0;

    for(size_t i = 0; i < iteration_no; i ++) {
        model->first_oracle(data, sub_grad, true, &generator, &distribution);
        //FIXME: Modify step_size.
        double step_size = 1.0 / 10.0;

        _m ++;
        if(_m < m) {
            double* temp_sg = new double[MAX_DIM];
            copy_vec(temp_sg, sub_grad);
            double l2_sg = comp_l2_norm(temp_sg);
            for(size_t j = 0; j < MAX_DIM; j ++) {
                temp_sg[j] /= l2_sg;
                k_grad[j] += temp_sg[j];
            }
            delete[] temp_sg;
        }
        else {
            variety = comp_l2_norm(k_grad);
            printf("variety : %lf \n", variety);
            //variety = _param2 * pow(variety, 1.0 / _param1);
            for(size_t j = 0; j < MAX_DIM; j ++) {
                k_grad[j] = 0.0;
            }
            _m = 0;
        }

        for(size_t j = 0; j < MAX_DIM; j ++) {
            new_weights[j] = (model->get_model())[j] - step_size  * sub_grad[j];
        }
        model->update_model(new_weights);
        if(is_debug_mode) {
            double log_F = log(model->zero_oracle(data));
            printf("KGD: Iteration %zd, log_F: %lf.\n", i, log_F);
        }
        // else
        //     printf("KGD: Iteration %zd , variety: %lf.\n", i, variety);
        if(is_store_weight) {
            for(size_t j = 0; j < MAX_DIM; j ++) {
                stored_weights[i * MAX_DIM + j] = new_weights[j];
            }
        }
    }
    //Final Output
    double log_F = log(model->zero_oracle(data));
    printf("KGD: Iteration %zd, log_F: %lf.\n", iteration_no, log_F);
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
    //FIXME: Epoch Size
    double m0 = (double) data->size() * 2.0;
    size_t total_iterations = 0;
    //OUTTER_LOOP
    for(size_t i = 0 ; i < outter_iteration_no; i ++) {
        model->first_oracle(data, full_grad);
        //FIXME:
        double inner_m = m0;//pow(2, i + 1) * m0;
        double* new_weights = new double[MAX_DIM];
        copy_vec(inner_weights, model->get_model());
        //INNER_LOOP
        for(size_t j = 0; j < inner_m ; j ++) {
            int rand_samp = distribution(generator);
            model->first_oracle(data, inner_sub_grad, rand_samp, inner_weights);
            model->first_oracle(data, outter_sub_grad, rand_samp);
            for(size_t k = 0; k < MAX_DIM; k ++) {
                vr_sub_grad[k] = inner_sub_grad[k] - outter_sub_grad[k] + full_grad[k];
            }
            //FIXME: Choose Step Size, Select weight scheme
            double step_size = 1.0 / 5.0;
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
            total_iterations ++;
            if(is_debug_mode) {
                double log_F = log(model->zero_oracle(data, inner_weights));
                printf("SVRG: Outter Iteration: %zd -> Inner Iteration %zd, log_F for inner_weights: %lf.\n", i, j, log_F);
            }
            else
                printf("SVRG: Iteration %zd.\n", total_iterations);
        }
        //FIXME: Different Update Options.
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
    //Final Output
    double log_F = log(model->zero_oracle(data));
    printf("SVRG: Total Iteration No.: %zd, logF = %lf.\n", total_iterations, log_F);
    if(is_store_weight)
        return stored_weights;
    return NULL;
}

std::vector<double>* grad_desc::Katyusha(Data* data, blackbox* model, size_t outter_iteration_no
    , bool is_store_weight, bool is_debug_mode) {
    //Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, data->size() - 1);
    size_t m = 2.0 * data->size();
    size_t total_iterations = 0;
    //FIXME: Adjust Factors. (Lipschitz Constant)
    double tau_2 = 0.5, tau_1 = 0.1, L = 5.0, sigma = 0.0001;
    double alpha = 1.0 / (tau_1 * 3.0 * L);
    double step_size_y = 1.0 / (3.0 * L);
    double compos_factor = 1 + alpha * sigma;
    double compos_base = 1 - (compos_factor - pow(compos_factor, m)) / (alpha * sigma);
    double* y = new double[MAX_DIM];
    double* z = new double[MAX_DIM];
    double* inner_weights = new double[MAX_DIM];
    double* full_grad = new double[MAX_DIM];
    //init vectors
    copy_vec(y, model->get_model());
    copy_vec(z, model->get_model());
    copy_vec(inner_weights, model->get_model());
    //OUTTER LOOP
    for(size_t i = 0; i < outter_iteration_no; i ++) {
        model->first_component_oracle(data, full_grad);
        double* katyusha_grad = new double[MAX_DIM];
        double* inner_grad = new double[MAX_DIM];
        double* outter_grad = new double[MAX_DIM];
        double* new_weights = new double[MAX_DIM];
        //INNER LOOP
        for(size_t j = 0; j < m; j ++) {
            for(size_t k = 0; k < MAX_DIM; k ++) {
                inner_weights[k] = tau_1 * z[k] + tau_2 * (model->get_model())[k]
                                 + (1 - tau_1 - tau_2) * y[k];
            }
            int rand_samp = distribution(generator);
            model->first_component_oracle(data, inner_grad, rand_samp, inner_weights);
            model->first_component_oracle(data, outter_grad, rand_samp);
            for(size_t k = 0; k < MAX_DIM; k ++) {
                katyusha_grad[k] = full_grad[k] + inner_grad[k] - outter_grad[k];
            }
            for(size_t k = 0; k < MAX_DIM; k ++) {
                z[k] -= alpha * katyusha_grad[k];
            }
            model->proximal_regularizer(z, alpha);
            for(size_t k = 0; k < MAX_DIM; k ++) {
                y[k] = inner_weights[k] - step_size_y * katyusha_grad[k];
            }
            model->proximal_regularizer(y, step_size_y);
            for(size_t k = 0; k < MAX_DIM; k ++) {
                new_weights[k] += pow(compos_factor, j) / compos_base * y[k];
            }
            total_iterations ++;
            if(is_debug_mode) {
                double log_F = log(model->zero_oracle(data, inner_weights));
                printf("Katyusha: Outter Iteration: %zd -> Inner Iteration %zd, log_F for inner_weights: %lf.\n", i, j, log_F);
            }
            else
                printf("Katyusha: Iteration %zd.\n", total_iterations);
        }
        model->update_model(new_weights);
        delete[] katyusha_grad;
        delete[] inner_grad;
        delete[] outter_grad;
        delete[] new_weights;
        if(is_debug_mode) {
            double log_F = log(model->zero_oracle(data));
            printf("Katyusha: Outter Iteration %zd, log_F: %lf.\n", i, log_F);
        }
    }
    delete[] y;
    delete[] z;
    delete[] inner_weights;
    delete[] full_grad;
    //Final Output
    double log_F = log(model->zero_oracle(data));
    printf("Katyusha: Total Iteration No.: %zd, log_F: %lf.\n", total_iterations, log_F);
    return NULL;
}

std::vector<double>* grad_desc::SAG(Data* data, blackbox* model, bool is_store_weight
    , bool is_debug_mode) {
    //TODO:
    return NULL;
}

std::vector<double>* grad_desc::SAGA(Data* data, blackbox* model, bool is_store_weight
    , bool is_debug_mode) {
    //TODO:
    return NULL;
}
