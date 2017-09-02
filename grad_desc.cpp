#include "grad_desc.hpp"
#include "utils.hpp"
#include <random>
#include <math.h>
#include <string.h>

extern size_t MAX_DIM;

void debug(std::string s, double a) {
    if(a != a) {
        throw s + " " + std::to_string(a);
    }
}

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
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, data->size() - 1);
    double* stored_F = NULL;
    size_t passes = (size_t) floor((double) iteration_no / data->size());
    double* stored_weights = NULL;
    // lazy updates extra array.
    int* last_seen = new int[MAX_DIM];
    for(size_t i = 0; i < MAX_DIM; i ++) last_seen[i] = -1;
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

std::vector<double>* grad_desc::Prox_SVRG(Data* data, blackbox* model, size_t& iteration_no, int Mode
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
    //FIXME: Epoch Size(SVRG / SVRG++)
    double m0 = (double) N * 2.0;
    size_t total_iterations = 0;
    copy_vec(inner_weights, model->get_model());
    // OUTTER_LOOP
    for(size_t i = 0 ; i < iteration_no; i ++) {
        double* full_grad_core = new double[N];
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
                full_grad[index] += iter.next() * full_grad_core[j] / (double) N;
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
        for(size_t j = 0; j < inner_m; j ++) {
            int rand_samp = distribution(generator);
            double inner_core = model->first_component_oracle_core(data, rand_samp, inner_weights);
            for(Data::iterator iter = (*data)(rand_samp); iter.hasNext();) {
                size_t index = iter.getIndex();
                double val = iter.next();
                // lazy update
                if((int)j > last_seen[index] + 1) {
                    aver_weights[index] += model->proximal_regularizer(inner_weights[index]
                        , step_size, j - (last_seen[index] + 1), -step_size * full_grad[index]) / inner_m;
                }
                double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val + full_grad[index];
                inner_weights[index] -= step_size * vr_sub_grad;
                aver_weights[index] += model->proximal_regularizer(inner_weights[index], step_size) / inner_m;
                last_seen[index] = j;
            }
            total_iterations ++;
            // For Drawing
            if(is_store_weight) {
                for(size_t k = 0; k < MAX_DIM; k ++)
                    stored_weights->push_back(inner_weights[k]);
            }
            if(is_debug_mode) {
                double log_F = log(model->zero_oracle(data, inner_weights));
                printf("Prox_SVRG: Outter Iteration: %zd -> Inner Iteration %zd, log_F for inner_weights: %lf.\n", i, j, log_F);
            }
        }
        // lazy update aggragate
        for(size_t j = 0; j < MAX_DIM; j ++) {
            if(inner_m > last_seen[j] + 1) {
                aver_weights[j] += model->proximal_regularizer(inner_weights[j], step_size
                    , inner_m - (last_seen[j] + 1), -step_size * full_grad[j]) / inner_m;
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
        // For Matlab (per m/n passes)
        if(is_store_result) {
            stored_F->push_back(model->zero_oracle(data));
        }
        delete[] last_seen;
        delete[] aver_weights;
        delete[] full_grad_core;
        if(is_debug_mode) {
            double log_F = log(model->zero_oracle(data));
            printf("Prox_SVRG: Outter Iteration %zd, log_F: %lf.\n", i, log_F);
        }
    }
    delete[] full_grad;
    delete[] inner_weights;
    //Final Output
    double log_F = log(model->zero_oracle(data));
    iteration_no = total_iterations;
    printf("Prox_SVRG: Total Iteration No.: %zd, logF = %lf.\n", total_iterations, log_F);
    if(is_store_weight)
        return stored_weights;
    if(is_store_result)
        return stored_F;
    return NULL;
}

double equal_ratio(double p, double pow_term) {
    return p * (1 - pow_term) / (1 - p);
}

// Magic Code
double Katyusha_Y_L2_proximal(double& _y0, double _z0, double tau_1, double tau_2
    , double lambda, double step_size_y, double alpha, double _outterx, double _F
    , size_t times, int start_iter, double compos_factor, double compos_base) {
    double _factor = (1 - tau_1 - tau_2);
    double A = _factor / (1 + step_size_y * lambda);
    double S = _z0 + _F / lambda;
    double Alpha = 1 / (1 + lambda * alpha);
    double Const = (tau_2 * _outterx - step_size_y * _F - tau_1 * _F / lambda)
                 / _factor;
    double Quo = A / Alpha;
    double pow_A = pow((double)A, (double)times);
    double pow_compos_factor = pow((double)compos_factor, (double)times);
    double pow_Alpha = pow((double)Alpha, (double)times);
    // Lazy Average
    double start_pos = pow((double)compos_factor, (double)start_iter) / compos_base;
    double t_P1 = compos_factor * A;
    double P1 = start_pos * _y0 * equal_ratio(t_P1, pow_compos_factor * pow_A);
    double t_P2 = tau_1 / (_factor * Alpha) * S * start_pos * A / (1 - A / Alpha);
    double t2_P2 = equal_ratio(Alpha * compos_factor, pow_Alpha * pow_compos_factor)
                 - equal_ratio(t_P1, pow_compos_factor * pow_A);
    double P2 = t_P2 * t2_P2;
    double t_P3 = start_pos * A / (1 - A) * Const;
    double t2_P3 = equal_ratio(compos_factor , pow_compos_factor)
                 - equal_ratio(A * compos_factor, pow_A * pow_compos_factor);
    double P3 = t_P3 * t2_P3;
    double lazy_average = P1 + P2 + P3;
    // Proximal update K times
    _y0 = pow_A * _y0 + tau_1 / _factor * S * pow((double)Alpha , (double)(times - 1)) * A
    * (1 - pow((double)Quo, (double)times)) / (1 - Quo) + equal_ratio(A, pow_A) * Const;
    return lazy_average;
}

std::vector<double>* grad_desc::Katyusha(Data* data, blackbox* model, size_t& iteration_no, double L
    , double sigma, double step_size, bool is_store_weight, bool is_debug_mode, bool is_store_result) {
    // Random Generator
    std::vector<double>* stored_F = new std::vector<double>;
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, data->size() - 1);
    size_t m = 2.0 * data->size();
    size_t N = data->size();
    size_t total_iterations = 0;
    double lambda = model->get_param(0);
    double tau_2 = 0.5, tau_1 = 0.4999;
    if(sqrt(sigma * m / (3.0 * L)) < 0.5) tau_1 = sqrt(sigma * m / (3.0 * L));
    double alpha = 1.0 / (tau_1 * 3.0 * L);
    double step_size_y = 1.0 / (3.0 * L);
    double compos_factor = 1 + alpha * sigma;
    double compos_base = 1 - (compos_factor - pow((double)compos_factor, (double)m)) / (alpha * sigma);
    double* compos_weights = new double[m];
    for(size_t i = 0; i < m; i ++)
        compos_weights[i] = pow((double)compos_factor, (double)i) / compos_base;
    double* y = new double[MAX_DIM];
    double* z = new double[MAX_DIM];
    double* inner_weights = new double[MAX_DIM];
    double* full_grad = new double[MAX_DIM];
    // init vectors
    copy_vec(y, model->get_model());
    copy_vec(z, model->get_model());
    copy_vec(inner_weights, model->get_model());
    // OUTTER LOOP
    for(size_t i = 0; i < iteration_no; i ++) {
        double* full_grad_core = new double[N];
        double* outter_weights = (model->get_model());
        double* aver_weights = new double[MAX_DIM];
        // lazy update extra param
        double* last_seen = new double[MAX_DIM];
        memset(full_grad, 0, MAX_DIM * sizeof(double));
        memset(aver_weights, 0, MAX_DIM * sizeof(double));
        for(size_t i = 0; i < MAX_DIM; i ++) last_seen[i] = -1;
        // Full Gradient
        for(size_t j = 0; j < N; j ++) {
            full_grad_core[j] = model->first_component_oracle_core(data, j);
            for(Data::iterator iter = (*data)(j); iter.hasNext();) {
                size_t index = iter.getIndex();
                full_grad[index] += iter.next() * full_grad_core[j] / (double) N;
            }
        }
        // 0th Inner Iteration
        for(size_t k = 0; k < MAX_DIM; k ++)
            inner_weights[k] = tau_1 * z[k] + tau_2 * outter_weights[k]
                             + (1 - tau_1 - tau_2) * y[k];
        // INNER LOOP
        for(size_t j = 0; j < m; j ++) {
            int rand_samp = distribution(generator);
            double inner_core = model->first_component_oracle_core(data, rand_samp, inner_weights);
            for(Data::iterator iter = (*data)(rand_samp); iter.hasNext(); ) {
                size_t index = iter.getIndex();
                double val = iter.next();
                // lazy update
                if((int)j > last_seen[index] + 1) {
                    aver_weights[index] += Katyusha_Y_L2_proximal(y[index], z[index]
                        , tau_1, tau_2, lambda, step_size_y, alpha, outter_weights[index]
                        , full_grad[index], j - (last_seen[index] + 1), last_seen[index]
                        , compos_factor, compos_base);
                    model->proximal_regularizer(z[index], alpha, j - (last_seen[index] + 1)
                         , -alpha * full_grad[index]);
                    inner_weights[index] = tau_1 * z[index] + tau_2 * outter_weights[index]
                                     + (1 - tau_1 - tau_2) * y[index];
                }
                double katyusha_grad = full_grad[index] + val * (inner_core - full_grad_core[rand_samp]);
                z[index] -= alpha * katyusha_grad;
                model->proximal_regularizer(z[index], alpha);
                y[index] = inner_weights[index] - step_size_y * katyusha_grad;
                model->proximal_regularizer(y[index], step_size_y);
                aver_weights[index] += compos_weights[j] * y[index];
                // (j + 1)th Inner Iteration
                if(j < m - 1)
                    inner_weights[index] = tau_1 * z[index] + tau_2 * outter_weights[index]
                                     + (1 - tau_1 - tau_2) * y[index];
                last_seen[index] = j;
            }
            total_iterations ++;
            if(is_debug_mode) {
                double log_F = log(model->zero_oracle(data, inner_weights));
                printf("Katyusha: Outter Iteration: %zd -> Inner Iteration %zd, log_F for inner_weights: %lf.\n", i, j, log_F);
            }
        }
        // lazy update aggragate
        for(size_t j = 0; j < MAX_DIM; j ++) {
            if(m > last_seen[j] + 1) {
                aver_weights[j] += Katyusha_Y_L2_proximal(y[j], z[j]
                    , tau_1, tau_2, lambda, step_size_y, alpha, outter_weights[j]
                    , full_grad[j], m - (last_seen[j] + 1), last_seen[j]
                    , compos_factor, compos_base);
                model->proximal_regularizer(z[j], alpha, m - (last_seen[j] + 1)
                     , -alpha * full_grad[j]);
                inner_weights[j] = tau_1 * z[j] + tau_2 * outter_weights[j]
                                 + (1 - tau_1 - tau_2) * y[j];
            }
        }
        model->update_model(aver_weights);
        delete[] aver_weights;
        delete[] full_grad_core;
        delete[] last_seen;
        // For Matlab
        if(is_store_result) {
            stored_F->push_back(model->zero_oracle(data));
        }
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
