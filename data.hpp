#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <vector>

class Data {
public:
    Data(int N);
    void Increase(int M);
    //data
    double& operator()(size_t n, size_t d);
    double operator()(size_t n, size_t d) const;
    //label
    int& operator[](size_t n);
    int operator[](size_t n) const;
    size_t size() const;
private:
    size_t mN;
    std::vector<double> mData;
    std::vector<int> mLabel;
    void boundChecking(size_t n, size_t d) const;
};

#endif
