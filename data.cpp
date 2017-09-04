#include "data.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string>
extern size_t MAX_DIM;

void Data::boundChecking(size_t n, size_t d) const {
    if(d >= MAX_DIM || n >= size()) {
        throw std::string("500 Bound Overflow.");
    }
}

Data::Data(size_t N): mN(N), mData(N * MAX_DIM), mLabel(N), iter(*this), is_sparse(false) {}

// Matlab Mat Support
Data::Data(size_t N, double* Xt, double* Y, bool isSparse, size_t* Jc, size_t* Ir)
    :iter(*this), is_sparse(isSparse), jc(Jc), ir(Ir) {
    mN = N;
    std::vector<double> data(Xt, Xt + jc[N]);
    mData = data;
    std::vector<double> label(Y, Y + N);
    mLabel = label;
}

void Data::Increase(int M) {
    mN += M;
    mData.resize(mN * MAX_DIM);
    mLabel.resize(mN);
}

Data::iterator& Data::operator()(size_t n) {
    iter.reset(n);
    return iter;
}

double& Data::operator()(size_t n, size_t d) {
    if(!is_sparse) {
        boundChecking(n, 0);
        return mData[n * MAX_DIM + d];
    }
    else {
        throw std::string("405 Operator(n,d) does not support sparse data.");
    }
}

double Data::operator()(size_t n, size_t d) const {
    if(!is_sparse) {
        boundChecking(n, 0);
        return mData[n * MAX_DIM + d];
    }
    else {
        throw std::string("405 Operator(n,d) does not support sparse data.");
    }
}

double& Data::operator[](size_t n) {
    boundChecking(n, 0);
    return mLabel[n];
}

double Data::operator[](size_t n) const {
    boundChecking(n, 0);
    return mLabel[n];
}

size_t Data::size() const {
    return mN;
}

bool Data::issparse() const {
    return is_sparse;
}

void Data::iterator::reset(size_t n) {
    if(data.is_sparse)
        m_index = data.jc[n];
    else
        m_index = n * MAX_DIM;
    m_samp_index = n;
}

bool Data::iterator::hasNext() const {
    if(data.is_sparse && m_index >= data.jc[m_samp_index + 1])
        return false;
    else if(!data.is_sparse && m_index - m_samp_index * MAX_DIM >= MAX_DIM)
        return false;
    else
        return true;
}

double Data::iterator::next() {
    if(hasNext()) {
        return data.mData[m_index ++];
    }
    else {
        throw std::string("500 Data Bound Overflow.");
    }
}

size_t Data::iterator::getIndex() const {
    if(hasNext()) {
        if(data.is_sparse)
            return data.ir[m_index];
        else
            return m_index - m_samp_index * MAX_DIM;
    }
}
