#include "data.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string>
extern size_t MAX_DIM;

void Data::boundChecking(size_t n, size_t d) const {
    if(d >= MAX_DIM || n >= size()) {
        std::string err = "Bound Overflow.";
        throw err;
    }
}

Data::Data(int N): mN(N), mData(N * MAX_DIM), mLabel(N) {}

void Data::Increase(int M) {
    mN += M;
    mData.resize(mN * MAX_DIM);
    mLabel.resize(mN);
}

double& Data::operator()(size_t n, size_t d) {
    boundChecking(n, d);
    return mData[n * MAX_DIM + d];
}

double Data::operator()(size_t n, size_t d) const {
    boundChecking(n, d);
    return mData[n * MAX_DIM + d];
}

int& Data::operator[](size_t n) {
    boundChecking(n, 0);
    return mLabel[n];
}

int Data::operator[](size_t n) const {
    boundChecking(n, 0);
    return mLabel[n];
}

size_t Data::size() const {
    return mN;
}
