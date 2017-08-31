#include <iostream>
#include "mex.h"
#include "data.hpp"
#include "grad_desc.hpp"
#include "svm.hpp"
#include "regularizer.hpp"
#include "logistic.hpp"
#include "least_square.hpp"
#include "utils.hpp"

#include <sys/time.h>

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
        // :TIMING TEST
        struct timeval tp;
        gettimeofday(&tp, NULL);
        long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
        Data* data = parse_data(prhs, (bool) prhs[10]);
        gettimeofday(&tp, NULL);
        printf("Data Parse time: %ld ms\n", tp.tv_sec * 1000 + tp.tv_usec / 1000 - ms);

        double *init_weight = mxGetPr(prhs[5]);
        double lambda = mxGetScalar(prhs[6]);
        double L = mxGetScalar(prhs[7]);
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
        switch(_hash(_regul)) {
            case _hash("L2"):
                regularizer = regularizer::L2;
                break;
            case _hash("L1"):
                regularizer = regularizer::L1;
                break;
            default:
                mexErrMsgTxt("Unrecognized regularizer.");
                break;
        }

        blackbox* model;
        char* _model = new char[MAX_PARAM_STR_LEN];
        mxGetString(prhs[3], _model, MAX_PARAM_STR_LEN);
        switch(_hash(_model)) {
            case _hash("logistic"):
                model = new logistic(lambda, regularizer);
                break;
            case _hash("least_square"):
                model = new least_square(lambda, regularizer);
                break;
            case _hash("svm"):
                model = new svm(lambda, regularizer);
                break;
            default:
                mexErrMsgTxt("Unrecognized model.");
                break;
        }
        model->set_init_weights(init_weight);

        // TIMING TEST
        gettimeofday(&tp, NULL);
        ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;

        char* _algo = new char[MAX_PARAM_STR_LEN];
        mxGetString(prhs[2], _algo, MAX_PARAM_STR_LEN);
        double* stored_F;
        std::vector<double>* vec_stored_F;
        size_t len_stored_F;
        switch(_hash(_algo)) {
            case _hash("GD"):
                stored_F = grad_desc::GD(data, model, iteration_no, L, step_size,
                    false, false, is_store_result);
                len_stored_F = iteration_no;
                break;
            case _hash("SGD"):
                stored_F = grad_desc::SGD(data, model, iteration_no, L, step_size,
                    false, false, is_store_result);
                len_stored_F = (size_t) floor((double) iteration_no / data->size());
                break;
            case _hash("Prox_SVRG"):
                vec_stored_F = grad_desc::Prox_SVRG(data, model, iteration_no, Mode, L, step_size,
                    false, false, is_store_result);
                stored_F = &(*vec_stored_F)[0];
                len_stored_F = vec_stored_F->size();
                break;
            case _hash("SVRG"):
                if(regularizer == regularizer::L1)
                    mexErrMsgTxt("SVRG not applicable to L1 regularizer.");
                vec_stored_F = grad_desc::SVRG(data, model, iteration_no, Mode, L, step_size,
                    false, false, is_store_result);
                stored_F = &(*vec_stored_F)[0];
                len_stored_F = vec_stored_F->size();
                break;
            case _hash("Katyusha"):
                vec_stored_F = grad_desc::Katyusha(data, model, iteration_no, L, step_size,
                    false, false, is_store_result);
                stored_F = &(*vec_stored_F)[0];
                len_stored_F = vec_stored_F->size();
                break;
            default:
                mexErrMsgTxt("Unrecognized algorithm.");
                break;
        }
        //TIMING TEST
        gettimeofday(&tp, NULL);
        printf("Iteration time: %ld ms\n", tp.tv_sec * 1000 + tp.tv_usec / 1000 - ms);

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
