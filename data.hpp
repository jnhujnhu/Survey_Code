#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <vector>
#include <stdint.h>

class Data {
public:
    Data(size_t N);
    Data(size_t N, double* Xt, double* Y, bool isSparse = false, size_t* Jc = NULL,
        size_t* Ir = NULL);
    void Increase(int M);
    class iterator {
    public:
        iterator(Data& d):data(d) {}
        bool hasNext() const;
        size_t getIndex() const;
        double next();
        void reset(size_t n);
    private:
        Data& data;
        size_t m_index;
        size_t m_samp_index;
    };
    //data
    iterator& operator()(size_t n); //Sparse Support
    double& operator()(size_t n, size_t d);
    double operator()(size_t n, size_t d) const;
    //label
    double& operator[](size_t n);
    double operator[](size_t n) const;
    size_t size() const;
private:
    size_t mN;
    std::vector<double> mData;
    std::vector<double> mLabel;
    iterator iter;
    bool is_sparse;
    // Matlab Sparse Index
    size_t *jc, *ir;
    void boundChecking(size_t n, size_t d) const;
};

#endif
