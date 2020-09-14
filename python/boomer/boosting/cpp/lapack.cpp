#include "lapack.h"
#include <string>
#include <stdexcept>


Lapack::Lapack(dsysv_t dsysvFunction) {
    dsysvFunction_ = dsysvFunction;
}

int Lapack::queryDsysvLworkParameter(float64* tmpArray1, float64* output, int n) {
    // "U" if the upper-right triangle of A should be used, "L" if the lower-left triangle should be used
    char* uplo = "U";
    // The number of right-hand sides, i.e, the number of columns of the matrix B
    int nrhs = 1;
    // Set "lwork" parameter to -1, which indicates that the optimal value should be queried
    int lwork = -1;
    // Variable to hold the queried value
    double worksize;
    // Variable to hold the result of the solver. Will be 0 when terminated successfully, unlike 0 otherwise
    int info;

    // Query the optimal value for the "lwork" parameter...
    dsysvFunction_(uplo, &n, &nrhs, tmpArray1, &n, (int*) 0, output, &n, &worksize, &lwork, &info);

    if (info != 0) {
        throw std::runtime_error(
            std::string("DSYSV terminated with non-zero info code when querying the optimal lwork parameter: "
            + std::to_string(info)));
    }

    return (int) worksize;
}

void Lapack::dsysv(const float64* coefficients, const float64* invertedOrdinates, float64* tmpArray1, int* tmpArray2,
                   double* tmpArray3, float64* output, int n, int lwork, float64 l2RegularizationWeight) {
    // Copy the values in the arrays `invertedOrdinates` and `coefficients` to the arrays `output` and `tmpArray1`,
    // respectively...
    int i = 0;

    for (int c = 0; c < n; c++) {
        output[c] = -invertedOrdinates[c];
        int offset = c * n;

        for (int r = 0; r < c + 1; r++) {
            float64 tmp = coefficients[i];

            if (r == c) {
                tmp += l2RegularizationWeight;
            }

            tmpArray1[offset + r] = tmp;
            i++;
        }
    }

    // "U" if the upper-right triangle of A should be used, "L" if the lower-left triangle should be used
    char* uplo = "U";
    // The number of right-hand sides, i.e, the number of columns of the matrix B
    int nrhs = 1;
    // Variable to hold the result of the solver. Will be 0 when terminated successfully, unlike 0 otherwise
    int info;

    // Run the DSYSV solver...
    dsysvFunction_(uplo, &n, &nrhs, tmpArray1, &n, tmpArray2, output, &n, tmpArray3, &lwork, &info);

    if (info != 0) {
        throw std::runtime_error(std::string("DSYSV terminated with non-zero info code: " + std::to_string(info)));
    }
}
