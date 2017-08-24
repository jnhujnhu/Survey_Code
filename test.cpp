#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include "data.hpp"
#include "grad_desc.hpp"
#include "svm.hpp"
#include "regularizer.hpp"
#include "logistic.hpp"
#include "least_square.hpp"
#include "utils.hpp"
#include <sys/time.h>
#include <random>
using namespace std;

size_t MAX_DIM = 123;

Data* parse_data(char* data_dir) {
    Data* data = new Data(0);
    ifstream in(data_dir, ios::in);
    string line;
    while(getline(in, line)) {
        int label = stoi(line.substr(0, line.find(" ")));
        string dims_str = line.substr(line.find(" ") + 1);
        vector<string> dims = split(dims_str, ":1 ");
        data->Increase(1);
        for(size_t i = 0; i < dims.size(); i ++) {
            (*data)(data->size() - 1, (uint8_t) stoi(dims[i]) - 1) = 1;
        }
        (*data)[data->size() - 1] = label;
    }
    return data;
}

// For train(2d).dat, n = 2
Data* parse_data_2(char* data_dir) {
    freopen(data_dir, "r", stdin);
    int mN;
    scanf("%d", &mN);
    Data* data = new Data(mN);
    double x, y, z;
    int label, _index = 0;
    char _no[30];
    while(scanf("%s %lf %lf %lf %d", _no, &x, &y, &z, &label) != EOF) {
        (*data)(_index, 0) = x;
        (*data)(_index, 1) = y;
        //data(_index, 2) = z;
        (*data)[_index ++] = label;
    }
    fclose(stdin);
    if(_index != mN) {
        string err = "Fake row number.";
        throw err;
    }
    return data;
}

//C linkage
extern "C" {
    size_t get_dim() {
        return MAX_DIM;
    }
    Data* get_data() {
        char* data_dir = (char*) "train(2d).dat";
        return parse_data_2(data_dir);
    }
    void Data_free(Data* data) {
        delete data;
    }
    svm* SVM_new() {
        return new svm(0.0001);
    }
    least_square* RLS_new() {
        least_square* rls = new least_square(0.01);
        double weight[2] = {20, -10};
        rls->set_init_weights(weight);
        return rls;
    }
    void Model_free(blackbox* model) {
        delete model;
    }
    double obj_func(blackbox* model, Data* data, double weight0, double weight1) {
        double weight[2] = {weight0, weight1};
        return log(model->zero_oracle(data, weight));
    }
    void Free(double* point) {
        delete[] point;
    }
    double* GD(blackbox* model, int step_no, Data* data) {
        double* stored_weights = grad_desc::GD(data, model, step_no, true, false);
        return stored_weights;
    }
    double* SGD(blackbox* model, int step_no, Data* data) {
        double* stored_weights = grad_desc::SGD(data, model, step_no, true, false);
        return stored_weights;
    }
    // double* KGD(blackbox* model, int step_no, Data* data) {
    //     std::vector<double>* stored_weights = grad_desc::KGD(data, model, step_no, true, false);
    //     return &(*stored_weights)[0];
    // }
    double* SVRG(blackbox* model, int step_no, Data* data) {
        std::vector<double>* stored_weights = grad_desc::SVRG(data, model, step_no, true, false);
        return &(*stored_weights)[0];
    }
}

// Simply estimate (more sophosticated: Using Reversed Weibull)
double evaluate_lipschitz_constant(Data* data, blackbox* model) {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution_x(-5, 5);
    double _lipschitz = 0.0;
    double scaling_factor = 1.0;
    for(size_t i = 0; i < 10000; i ++) {
        double* x = new double[MAX_DIM];
        double* y = new double[MAX_DIM];
        for(size_t j = 0; j < MAX_DIM; j ++) {
            x[j] = distribution_x(generator);
            //std::uniform_real_distribution<double> distribution_y(x[j] - 0.01, x[j] + 0.01);
            y[j] = distribution_x(generator);
        }
        double *_pfy = new double[MAX_DIM];
        double *_pfx = new double[MAX_DIM];
        model->first_component_oracle(data, _pfy, false, NULL, NULL, y);
        model->first_component_oracle(data, _pfx, false, NULL, NULL, x);
        for(size_t j = 0; j < MAX_DIM; j ++) {
            _pfx[j] = abs(_pfx[j] - _pfy[j]);
            x[j] = abs(x[j] - y[j]);
        }
        double _t_lipschitz = comp_l2_norm(_pfx) / comp_l2_norm(x);
        cout << i << " : " << _t_lipschitz << endl;
        if(_t_lipschitz > _lipschitz)
            _lipschitz = _t_lipschitz;
        delete[] x;
        delete[] y;
        delete[] _pfx;
        delete[] _pfy;
    }
    return _lipschitz * scaling_factor;
}

//Test Main
int main() {
    char* data_dir = (char*) "a9a";
    try {
<<<<<<< HEAD
        logistic* rls = new logistic(0.0001);
=======
        logistic* logis= new logistic(0.0001, regularizer::L1);
>>>>>>> 27a95fa117b1486ffda4fc7007886958482c0ff0
        Data* data = parse_data(data_dir);
        // double weight[2] = {200, -100};
        // rls->set_init_weights(weight);
        //cout << evaluate_lipschitz_constant(data, logis) << endl;
        grad_desc::KGD(data, rls, 500000, false, false);
        //grad_desc::SGD(data, rls, 1000, false, false);
        // for(size_t j = 0; j < MAX_DIM; j ++) {
        //     printf("%lf ", logis->get_model()[j]);
        // }
        // cout << endl;
        // :TIMING TEST
        // struct timeval tp;
        // gettimeofday(&tp, NULL);
        // long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
        // grad_desc::SVRG(data, logis, 10, false, true);
        // gettimeofday(&tp, NULL);
        // printf("Execuate time: %ld \n", tp.tv_sec * 1000 + tp.tv_usec / 1000 - ms);
        delete rls;
        delete data;
    } catch(string c) {
        cerr << c << endl;
        exit(EXIT_FAILURE);
    }
    return 0;
}
