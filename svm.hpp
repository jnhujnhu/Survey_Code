#ifndef NEW_SVM_H
#define NEW_SVM_H
#include "blackbox.hpp"
#include "regularizer.hpp"

class svm: public blackbox {
public:
    svm(double param, int regular = regularizer::L2);
    int classify(double* sample) const override;
    double zero_component_oracle(Data* data, double* weights = NULL) const override;
    double first_component_oracle_core(Data* data, int given_index, double* weights = NULL) const override;
};

#endif
