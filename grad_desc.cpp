#include "grad_desc.hpp"
#include "utils.hpp"
#include <random>
#include <math.h>
#include <string.h>

#include <sys/time.h>

extern size_t MAX_DIM;
double* grad_desc::GD(Data* data, blackbox* model, size_t iteration_no, double L, double step_size
    , bool is_store_weight, bool is_debug_mode, bool is_store_result) {
    double* stored_weights = NULL;
    double* stored_F = NULL;
    size_t passes = iteration_no;
    if(is_store_weight)
        stored_weights = new double[iteration_no * MAX_DIM];
    if(is_store_result)
        stored_F = new double[passes];
    double* new_weights = new double[MAX_DIM];
    for(size_t i = 0; i < iteration_no; i ++) {
        double* full_grad = new double[MAX_DIM];
        model->first_oracle(data, full_grad);
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
        // For Drawing
        if(is_store_weight) {
            for(size_t j = 0; j < MAX_DIM; j ++) {
                stored_weights[i * MAX_DIM + j] = new_weights[j];
            }
        }
        //For Matlab
        if(is_store_result) {
            stored_F[i] = model->zero_oracle(data);
        }
        delete[] full_grad;
    }
    //Final Output
    double log_F = log(model->zero_oracle(data));
    printf("GD: Iteration %zd, log_F: %lf.\n", iteration_no, log_F);
    delete[] new_weights;
    if(is_store_weight)
        return stored_weights;
    if(is_store_result)
        return stored_F;
    return NULL;
}

double* grad_desc::SGD(Data* data, blackbox* model, size_t iteration_no, double L, double step_size
    , bool is_store_weight, bool is_debug_mode, bool is_store_result) {
    //Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, data->size() - 1);
    double* stored_F = NULL;
    size_t passes = (size_t) floor((double) iteration_no / data->size());
    double* stored_weights = NULL;
    // lazy updates extra array.
    int* last_seen = new int[MAX_DIM];
    for(size_t i = 0; i < MAX_DIM; i ++)
        last_seen[i] = -1;
    // For Drawing
    if(is_store_weight)
        stored_weights = new double[iteration_no * MAX_DIM];
    // For Matlab
    if(is_store_result)
        stored_F = new double[passes];

    double* sub_grad = new double[MAX_DIM];
    double* new_weights = new double[MAX_DIM];
    size_t N = data->size();
    memset(new_weights, 0, MAX_DIM * sizeof(double));
    for(size_t i = 0; i < iteration_no; i ++) {
        int rand_samp = distribution(generator);
        double core = model->first_component_oracle_core(data, rand_samp, new_weights);
        for(Data::iterator iter = (*data)(rand_samp); iter.hasNext();) {
            size_t index = iter.getIndex();
            // lazy update.
            if(new_weights[index] != 0 && (int)i > last_seen[index] + 1) {
                model->proximal_regularizer(new_weights[index], step_size
                    , i - (last_seen[index] + 1));
            }
            new_weights[index] -= step_size * core * iter.next();
            model->proximal_regularizer(new_weights[index], step_size);
            last_seen[index] = i;
        }
        if(is_debug_mode) {
            double log_F = log(model->zero_oracle(data));
            printf("SGD: Iteration %zd, log_F: %lf.\n", i, log_F);
        }
        // For Drawing
        if(is_store_weight) {
            for(size_t j = 0; j < MAX_DIM; j ++) {
                stored_weights[i * MAX_DIM + j] = new_weights[j];
            }
        }
        // For Matlab
        if(is_store_result) {
            if(!(i % N)) {
                stored_F[(size_t) floor((double) i / N)] = model->zero_oracle(data, new_weights);
            }
        }
    }
    model->update_model(new_weights);
    //Final Output
    double log_F = log(model->zero_oracle(data));
    printf("SGD: Iteration %zd, log_F: %lf.\n", iteration_no, log_F);
    delete[] sub_grad;
    delete[] new_weights;
    delete[] last_seen;
    // For Drawing
    if(is_store_weight)
        return stored_weights;
    // For Matlab
    if(is_store_result)
        return stored_F;
    return NULL;
}

// Test Only
double* grad_desc::KGD(Data* data, blackbox* model, size_t iteration_no, double L, double step_size
    , bool is_store_weight, bool is_debug_mode, bool is_store_result) {
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


// std::vector<double>* grad_desc::SVRG(Data* data, blackbox* model, size_t& iteration_no, double L, double step_size
//     , bool is_store_weight, bool is_debug_mode, bool is_store_result) {
//     //Random Generator
//     std::random_device rd;
//     std::default_random_engine generator(rd());
//     std::uniform_int_distribution<int> distribution(0, data->size() - 1);
//     std::vector<double>* stored_weights = new std::vector<double>;
//     std::vector<double>* stored_F = new std::vector<double>;
//     double* inner_weights = new double[MAX_DIM];
//     double* vr_sub_grad = new double[MAX_DIM];
//     double* full_grad = new double[MAX_DIM];
//     double* inner_sub_grad = new double[MAX_DIM];
//     double* outter_sub_grad = new double[MAX_DIM];
//     //FIXME: Epoch Size
//     double m0 = (double) data->size() * 2.0;
//     size_t total_iterations = 0;
//     //OUTTER_LOOP
//     for(size_t i = 0 ; i < iteration_no; i ++) {
//         model->first_component_oracle(data, full_grad);
//         //FIXME:
//         double inner_m = m0;//pow(2, i + 1) * m0;
//         double* new_weights = new double[MAX_DIM];
//         memset(new_weights, 0, MAX_DIM * sizeof(double));
//         copy_vec(inner_weights, model->get_model());
//         //INNER_LOOP
//         for(size_t j = 0; j < inner_m ; j ++) {
//             int rand_samp = distribution(generator);
//             model->first_component_oracle(data, inner_sub_grad, rand_samp, inner_weights);
//             model->first_component_oracle(data, outter_sub_grad, rand_samp);
//             for(size_t k = 0; k < MAX_DIM; k ++) {
//                 vr_sub_grad[k] = inner_sub_grad[k] - outter_sub_grad[k] + full_grad[k];
//                 inner_weights[k] -= step_size * vr_sub_grad[k];
//                 model->proximal_regularizer(inner_weights[k], step_size);
//                 new_weights[k] += inner_weights[k] / inner_m;
//             }
//             total_iterations ++;
//             // For Drawing
//             if(is_store_weight) {
//                 for(size_t k = 0; k < MAX_DIM; k ++)
//                     stored_weights->push_back(inner_weights[k]);
//             }
//             // For Matlab
//             if(is_store_result) {
//                 if(!(total_iterations % data->size())) {
//                     stored_F->push_back(model->zero_oracle(data, inner_weights));
//                 }
//             }
//             if(is_debug_mode) {
//                 double log_F = log(model->zero_oracle(data, inner_weights));
//                 printf("SVRG: Outter Iteration: %zd -> Inner Iteration %zd, log_F for inner_weights: %lf.\n", i, j, log_F);
//             }
//         }
//         //FIXME: Different Update Options.
//         model->update_model(new_weights);
//         delete[] new_weights;
//         if(is_debug_mode) {
//             double log_F = log(model->zero_oracle(data));
//             printf("SVRG: Outter Iteration %zd, log_F: %lf.\n", i, log_F);
//         }
//     }
//     delete[] inner_sub_grad;
//     delete[] outter_sub_grad;
//     delete[] vr_sub_grad;
//     delete[] full_grad;
//     delete[] inner_weights;
//     //Final Output
//     double log_F = log(model->zero_oracle(data));
//     iteration_no = total_iterations;
//     printf("SVRG: Total Iteration No.: %zd, logF = %lf.\n", total_iterations, log_F);
//     if(is_store_weight)
//         return stored_weights;
//     if(is_store_result)
//         return stored_F;
//     return NULL;
// }


std::vector<double>* grad_desc::SVRG(Data* data, blackbox* model, size_t& iteration_no, double L, double step_size
    , bool is_store_weight, bool is_debug_mode, bool is_store_result) {
    //Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, data->size() - 1);
    std::vector<double>* stored_weights = new std::vector<double>;
    std::vector<double>* stored_F = new std::vector<double>;
    double* inner_weights = new double[MAX_DIM];
    double* full_grad = new double[MAX_DIM];
    // lazy update extra params.
    int* last_seen = new int[MAX_DIM];
    //FIXME: Epoch Size
    size_t N = data->size();
    double m0 = (double) N * 2.0;
    size_t total_iterations = 0;
    for(size_t i = 0; i < MAX_DIM; i ++)
        last_seen[i] = -1;
    //OUTTER_LOOP
    for(size_t i = 0 ; i < iteration_no; i ++) {
        double* full_grad_core = new double[N];
        memset(full_grad, 0, MAX_DIM * sizeof(double));
        for(size_t j = 0; j < N; j ++) {
            full_grad_core[j] = model->first_component_oracle_core(data, j);
            for(Data::iterator iter = (*data)(j); iter.hasNext();) {
                size_t index = iter.getIndex();
                full_grad[index] += iter.next() * full_grad_core[j] / (double) N;
            }
        }
        //FIXME: SVRG / SVRG++
        double inner_m = m0;//pow(2, i + 1) * m0;
        double* new_weights = new double[MAX_DIM];
        memset(new_weights, 0, MAX_DIM * sizeof(double));
        copy_vec(inner_weights, model->get_model());
        //INNER_LOOP
        for(size_t j = 0; j < inner_m ; j ++) {
            int rand_samp = distribution(generator);
            double inner_core = model->first_component_oracle_core(data, rand_samp, inner_weights);
            for(Data::iterator iter = (*data)(rand_samp); iter.hasNext();) {
                size_t index = iter.getIndex();
                double val = iter.next();
                // lazy update
                if((int)j > last_seen[index] + 1) {
                    new_weights[index] += model->proximal_regularizer(inner_weights[index]
                        , step_size, j - (last_seen[index] + 1), -step_size * full_grad[index]
                        , true) / inner_m;
                }
                double vr_sub_grad = inner_core * val - full_grad_core[rand_samp] * val + full_grad[index];
                inner_weights[index] -= step_size * vr_sub_grad;
                model->proximal_regularizer(inner_weights[index], step_size);
                last_seen[index] = j;
                // Average update option.
                new_weights[index] += inner_weights[index] / inner_m;
            }
            total_iterations ++;
            // For Drawing
            if(is_store_weight) {
                for(size_t k = 0; k < MAX_DIM; k ++)
                    stored_weights->push_back(inner_weights[k]);
            }
            // For Matlab
            if(is_store_result) {
                if(!(total_iterations % N)) {
                    stored_F->push_back(model->zero_oracle(data, inner_weights));
                }
            }
            if(is_debug_mode) {
                double log_F = log(model->zero_oracle(data, inner_weights));
                printf("SVRG: Outter Iteration: %zd -> Inner Iteration %zd, log_F for inner_weights: %lf.\n", i, j, log_F);
            }
        }
        // lazy update aggragate
        for(size_t j = 0; j < MAX_DIM; j ++) {
            if(inner_m > last_seen[j] + 1) {
                new_weights[j] += model->proximal_regularizer(inner_weights[j], step_size
                    , inner_m - (last_seen[j] + 1), -step_size * full_grad[j]
                    , true) / inner_m;
            }
        }
        //FIXME: Different Update Options.
        model->update_model(new_weights);
        delete[] new_weights;
        delete[] full_grad_core;
        if(is_debug_mode) {
            double log_F = log(model->zero_oracle(data));
            printf("SVRG: Outter Iteration %zd, log_F: %lf.\n", i, log_F);
        }
    }
    delete[] last_seen;
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

std::vector<double>* grad_desc::Katyusha(Data* data, blackbox* model, size_t& iteration_no, double L, double step_size
    , bool is_store_weight, bool is_debug_mode, bool is_store_result) {
    //Random Generator
    std::vector<double>* stored_F = new std::vector<double>;
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, data->size() - 1);
    size_t m = 2.0 * data->size();
    size_t total_iterations = 0;
    //FIXME: Adjust Factors. (Lipschitz Constant)
    double tau_2 = 0.5, tau_1 = 0.5, sigma = 0.0001;
    if(sqrt(sigma * m / 3.0 * L) < 0.5) tau_1 = sqrt(sigma * m / 3.0 * L);
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
    for(size_t i = 0; i < iteration_no; i ++) {
        model->first_component_oracle(data, full_grad);
        double* katyusha_grad = new double[MAX_DIM];
        double* inner_grad = new double[MAX_DIM];
        double* outter_grad = new double[MAX_DIM];
        double* new_weights = new double[MAX_DIM];
        memset(new_weights, 0, MAX_DIM * sizeof(double));
        memset(katyusha_grad, 0, MAX_DIM * sizeof(double));
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
            for(size_t j = 0; j < MAX_DIM; j ++)
                model->proximal_regularizer(z[j], alpha);
            for(size_t k = 0; k < MAX_DIM; k ++) {
                y[k] = inner_weights[k] - step_size_y * katyusha_grad[k];
            }
            for(size_t j = 0; j < MAX_DIM; j ++)
                model->proximal_regularizer(y[j], step_size_y);
            for(size_t k = 0; k < MAX_DIM; k ++) {
                new_weights[k] += pow(compos_factor, j) / compos_base * y[k];
            }
            total_iterations ++;
            // For Matlab
            if(is_store_result) {
                if(!(total_iterations % data->size())) {
                    stored_F->push_back(model->zero_oracle(data, inner_weights));
                }
            }
            if(is_debug_mode) {
                double log_F = log(model->zero_oracle(data, inner_weights));
                printf("Katyusha: Outter Iteration: %zd -> Inner Iteration %zd, log_F for inner_weights: %lf.\n", i, j, log_F);
            }
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
    iteration_no = total_iterations;
    printf("Katyusha: Total Iteration No.: %zd, log_F: %lf.\n", total_iterations, log_F);
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
