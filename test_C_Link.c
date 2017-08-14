#include <stdio.h>

extern void* SVM_new();
extern void* Data_new();
extern void SVM_free();
extern void Data_free();
extern void get_data(void* ptr);
extern double obj_func(void* svm, void* data, double a, double b);
extern void print(void* data);

int main() {
    void* data = Data_new();
    void* svm = SVM_new();
    get_data(data);
    print(data);
    //printf("%lf", obj_func(svm, data, 1.0, 1.0));
    SVM_free(svm);
    Data_free(data);
    return 0;
}
