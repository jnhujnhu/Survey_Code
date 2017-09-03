#include <iostream>
#include "mex.h"
#include "data.hpp"
#include "grad_desc.hpp"
#include "svm.hpp"
#include "regularizer.hpp"
#include "logistic.hpp"
#include "least_square.hpp"
#include "utils.hpp"
#include <string.h>

size_t MAX_DIM;

const size_t MAX_PARAM_STR_LEN = 15;

Data* parse_data(const mxArray* prhs[], bool is_sparse) {
    double *X = mxGetPr(prhs[0]);
    double *Y = mxGetPr(prhs[1]);
    size_t* jc;
    size_t* ir;
    MAX_DIM = mxGetM(prhs[0]);
    size_t N = mxGetN(prhs[0]);
    Data* data;
    if(is_sparse) {
        jc = mxGetJc(prhs[0]);
        ir = mxGetIr(prhs[0]);
        data = new Data(N, X, Y, is_sparse, jc, ir);
    }
    else
        data = new Data(N, X, Y);
    return data;
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    try {
        // Encapsulate Data
        Data* data = parse_data(prhs, (bool) prhs[10]);

        double *init_weight = mxGetPr(prhs[5]);
        double lambda = mxGetScalar(prhs[6]);
        double L = mxGetScalar(prhs[7]);
        double sigma = mxGetScalar(prhs[12]);
        double step_size = mxGetScalar(prhs[8]);
        // For SVRG
        int Mode = (int) mxGetScalar(prhs[11]);
        size_t iteration_no = (size_t) mxGetScalar(prhs[9]);
        bool is_store_result = false;
        if(nlhs == 1)
            is_store_result = true;

        int regularizer;
        char* _regul = new char[MAX_PARAM_STR_LEN];
        mxGetString(prhs[4], _regul, MAX_PARAM_STR_LEN);
        if(strcmp(_regul, "L2") == 0) {
            regularizer = regularizer::L2;
        }
        else if(strcmp(_regul, "L1") == 0){
            regularizer = regularizer::L1;
        }
        else mexErrMsgTxt("400 Unrecognized regularizer.");

        blackbox* model;
        char* _model = new char[MAX_PARAM_STR_LEN];
        mxGetString(prhs[3], _model, MAX_PARAM_STR_LEN);
        if(strcmp(_model, "logistic") == 0) {
                model = new logistic(lambda, regularizer);
        }
        else if(strcmp(_model, "least_square") == 0) {
            model = new least_square(lambda, regularizer);
        }
        else if(strcmp(_model, "svm") == 0) {
            model = new svm(lambda, regularizer);
        }
        else mexErrMsgTxt("400 Unrecognized model.");
        model->set_init_weights(init_weight);

        char* _algo = new char[MAX_PARAM_STR_LEN];
        mxGetString(prhs[2], _algo, MAX_PARAM_STR_LEN);
        double* stored_F;
        std::vector<double>* vec_stored_F;
        size_t len_stored_F;
        if(strcmp(_algo, "GD") == 0) {
            stored_F = grad_desc::GD(data, model, iteration_no, L, step_size,
                false, false, is_store_result);
            len_stored_F = iteration_no;
        }
        else if(strcmp(_algo, "SGD") == 0) {
            stored_F = grad_desc::SGD(data, model, iteration_no, L, step_size,
                false, false, is_store_result);
            len_stored_F = (size_t) floor((double) iteration_no / data->size());
        }
        else if(strcmp(_algo, "Prox_SVRG") == 0) {
            vec_stored_F = grad_desc::Prox_SVRG(data, model, iteration_no, Mode, L, step_size,
                false, false, is_store_result);
            stored_F = &(*vec_stored_F)[0];
            len_stored_F = vec_stored_F->size();
        }
        else if(strcmp(_algo, "SVRG") == 0) {
            if(regularizer == regularizer::L1)
                mexErrMsgTxt("405 SVRG not applicable to L1 regularizer.");
            vec_stored_F = grad_desc::SVRG(data, model, iteration_no, Mode, L, step_size,
                false, false, is_store_result);
            stored_F = &(*vec_stored_F)[0];
            len_stored_F = vec_stored_F->size();
        }
        else if(strcmp(_algo, "Katyusha") == 0) {
            if(regularizer == regularizer::L1)
                mexErrMsgTxt("405 Katyusha not applicable to L1 regularizer.");
            vec_stored_F = grad_desc::Katyusha(data, model, iteration_no, L, sigma, step_size,
                false, false, is_store_result);
            stored_F = &(*vec_stored_F)[0];
            len_stored_F = vec_stored_F->size();
        }
        else mexErrMsgTxt("400 Unrecognized algorithm.");

        if(is_store_result) {
            plhs[0] = mxCreateDoubleMatrix(len_stored_F, 1, mxREAL);
        	double* res_stored_F = mxGetPr(plhs[0]);
            for(size_t i = 0; i < len_stored_F; i ++)
                res_stored_F[i] = stored_F[i];
        }
        delete[] stored_F;
        delete model;
        delete data;
    } catch(std::string c) {
        std::cerr << c << std::endl;
        //exit(EXIT_FAILURE);
    }
}
