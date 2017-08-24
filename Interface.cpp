#include "mex.h"
#include "data.hpp"
#include "grad_desc.hpp"
#include "svm.hpp"
#include "regularizer.hpp"
#include "logistic.hpp"
#include "least_square.hpp"
#include "utils.hpp"

Data* parse_data(const mxArray* prhs[], bool is_sparse) {
    double *X = mxGetPr(prhs[0]);
    double *Y = mxGetPr(prhs[1]);
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    try {
        Data* data = parse_data(plhs, (bool) plhs[10]);

        double *init_weight = mxGetPr(prhs[5]);
        double lambda = mxGetScalar(prhs[6]);
        size_t iteration_no = (size_t) mxGetScalar(prhs[9]);
        bool is_store_weight = false;
        if(nlhs == 1)
            is_store_weight = true;

        int regularizer;
        char* _regul;
        mxGetString(prhs[4], _regul);
        std::string Regularizer(_regul);
        switch(_hash(Regularizer)) {
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
        char* _mode;
        mxGetString(prhs[3], _model);
        std::string Model(_model);
        switch(_hash(Model)) {
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

        char* _algo;
        mxGetString(prhs[2], _algo);
        std::string Algorithm(_algo);
        double* stored_weights;
        //TODO: Add step_size and L as params
        switch(_hash(Algorithm)) {
            case _hash("GD"):
                stored_weights = grad_desc::GD(data, model, iteration_no,
                    is_store_weight, false);
                break;
            case _hash("SGD"):
                stored_weights = grad_desc::SGD(data, model, iteration_no,
                    is_store_weight, false);
                break;
            case _hash("SVRG"):
                stored_weights = &(*grad_desc::SVRG(data, model, iteration_no,
                    is_store_weight, false))[0];
                break;
            case _hash("Katyusha"):
                //FIXME: Not support yet.
                stored_weights = &(*grad_desc::Katyusha(data, model, iteration_no,
                    is_store_weight, false))[0];
                break;
            default:
                mexErrMsgTxt("Unrecognized algorithm.");
                break;
        }
        //FIXME: Compute iter for 2-level algos.
        plhs[0] = mxCreateDoubleMatrix(iteration_no, 1, mxREAL);
    	double* res_stored_weights = mxGetPr(plhs[0]);
        for(size_t i = 0; i < iteration_no; i ++)
            res_stored_weights[i] = stored_weights[i];

        delete[] stored_weights;
        delete model;
        delete data;
    } catch(string c) {
        cerr << c << endl;
        exit(EXIT_FAILURE);
    }
}
