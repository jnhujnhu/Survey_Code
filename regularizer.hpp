#ifndef REGULARIZER_HPP
#define REGULARIZER_HPP

namespace regularizer {
    const int L2 = 1;
    const int L1 = 0;
    const int NONE = -1;
    double zero_oracle(int _regular, double lambda, double* weight);
    void first_oracle(int _regular, double* _pR, double lambda, double* weight);
    void proximal_operator(int _regular, double* _prox, double lambda, double step_size);
}
#endif
