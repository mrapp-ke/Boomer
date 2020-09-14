#include "blas.h"


Blas::Blas(ddot_t ddotFunction, dspmv_t dspmvFunction) {
    ddotFunction_ = ddotFunction;
    dspmvFunction_ = dspmvFunction;
}

float64 Blas::ddot(float64* x, float64* y, int n) {
    // Storage spacing between the elements of the arrays x and y
    int inc = 1;
    // Invoke the DDOT routine...
    return ddotFunction_(&n, x, &inc, y, &inc);
}

void Blas::dspmv(float64* a, float64* x, float64* output, int n) {
    // "U" if the upper-right triangle of A should be used, "L" if the lower-left triangle should be used
    char* uplo = "U";
    // A scalar to be multiplied with the matrix A
    double alpha = 1;
    // The increment for the elements of x and y
    int inc = 1;
    // A scalar to be multiplied with vector y
    double beta = 0;
    // Invoke the DSPMV routine...
    dspmvFunction_(uplo, &n, &alpha, a, x, &inc, &beta, output, &inc);
}
