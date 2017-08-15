#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include "data.hpp"
#include "grad_desc.hpp"
#include "svm.hpp"
#include <sys/time.h>
using namespace std;

size_t MAX_DIM = 2;

Data* parse_data(char* data_dir) {
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
        return parse_data(data_dir);
    }
    void Data_free(Data* data) {
        delete data;
    }
    svm* SVM_new() {
        return new svm(0.0001);
    }
    void SVM_free(svm* SVM) {
        delete SVM;
    }
    double obj_func(svm* SVM, Data* data, double weight0, double weight1) {
        double* weight = new double[2];
        weight[0] = weight0;
        weight[1] = weight1;
        return log(SVM->zero_oracle(data, weight));
    }
    void Free(double* point) {
        delete[] point;
    }
    double* SGD(svm* SVM, int step_no, Data* data) {
        double* stored_weights = grad_desc::SGD(data, SVM, step_no, true, false);
        return stored_weights;
    }
    double* SVRG(svm* SVM, int step_no, Data* data) {
        std::vector<double>* stored_weights = grad_desc::SVRG(data, SVM, step_no, true, false);
        return &(*stored_weights)[0];
    }
}

//Test Main
// int main() {
//     char* data_dir = (char*) "train(2d).dat";
//     try {
//         svm* SVM = new svm(0.0001);
//         Data* data = parse_data(data_dir);
//         grad_desc::SGD(data, SVM, 500000, false, true);
//         // :TIMING TEST
//         // struct timeval tp;
//         // gettimeofday(&tp, NULL);
//         // long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
//         //grad_desc::SVRG(data, SVM, 15, false, true);
//         // gettimeofday(&tp, NULL);
//         // printf("Execuate time: %ld \n", tp.tv_sec * 1000 + tp.tv_usec / 1000 - ms);
//         delete SVM;
//         delete data;
//     } catch(string c) {
//         cerr << c << endl;
//         exit(EXIT_FAILURE);
//     }
//     return 0;
// }
