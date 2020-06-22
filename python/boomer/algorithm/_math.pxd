# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides utility functions for common mathematical operations.
"""
from boomer.algorithm._arrays cimport intp, float64, array_float64, matrix_float64

from scipy.linalg.cython_lapack cimport dsysv
from scipy.linalg.cython_blas cimport dspmv, ddot
from libc.stdlib cimport malloc, free
from libc.math cimport pow


cdef inline divide_or_zero_float64(float64 a, float64 b):
    """
    Divides a number of dtype `float64` by another one. The division by zero evaluates to 0 by definition.

    :param a:   A scalar of dtype `float64`, representing the number to be divided
    :param b:   A scalar of dtype `float64`, representing the divisor
    :return:    A scalar of dtype `float64`, representing the result of a / b or 0, if b = 0
    """
    if b == 0:
        return 0
    else:
        return a / b


cdef inline intp triangular_number(intp n):
    """
    Computes and returns the n-th triangular number, i.e., the number of elements in a n times n triangle.

    :param n:   A scalar of dtype `intp`, representing the order of the triangular number
    :return:    A scalar of dtype `intp`, representing the n-th triangular number
    """
    return (n * (n + 1)) // 2


cdef inline l2_norm_pow(float64[::1] a):
    """
    Computes and returns the square of the L2 norm of a specific vector, i.e. the sum of the squares of its elements. To 
    obtain the actual L2 norm, the square-root of the result provided by this function must be computed.
    
    :param a:   An array of dtype `float64`, shape (n), representing a vector
    :return:    A scalar of dtype `float64`, representing the square of the L2 of the given vector
    """
    cdef float64 result = 0
    cdef intp n = a.shape[0]
    cdef float64 tmp
    cdef intp i

    for i in range(n):
        tmp = a[i]
        tmp = pow(tmp, 2)
        result += tmp

    return result


cdef inline float64 ddot_float64(float64[::1] x, float64[::1] y):
    """
    Computes and returns the dot product x * y of two vectors using BLAS' DDOT routine (see 
    http://www.netlib.org/lapack/explore-html/de/da4/group__double__blas__level1_ga75066c4825cb6ff1c8ec4403ef8c843a.html).
    
    :param x:   An array of dtype `float64`, shape (n), representing the first vector x
    :param y:   An array of dtype `float64`, shape (n), representing the second vector y
    :return:    A scalar of dtype `float64`, representing the result of the dot product x * y
    """
    # The number of elements in the arrays x and y
    cdef int n = x.shape[0]
    # Storage spacing between the elements of the arrays x and y
    cdef int inc = 1
    # Invoke the DDOT routine...
    cdef float64 result = ddot(&n, &x[0], &inc, &y[0], &inc)
    return result


cdef inline float64[::1] dspmv_float64(float64[::1] a, float64[::1] x):
    """
    Computes and returns the solution to the matrix-vector operation A * x using BLAS' DSPMV routine (see
    http://www.netlib.org/lapack/explore-html/d7/d15/group__double__blas__level2_gab746575c4f7dd4eec72e8110d42cefe9.html).
    This function expects A to be a double-precision symmetric matrix with shape `(n, n)` and x a double-precision array 
    with shape `(n)`.
    
    DSPMV expects the matrix A to be supplied in packed form, i.e., as an array with shape `(n * (n + 1) // 2 )` that 
    consists of the columns of A appended to each other and omitting all unspecified elements.
    
    :param a:   An array of dtype `float64`, shape `(n * (n + 1) // 2)`, representing the elements in the upper-right 
                triangle of the matrix A in a packed form
    :param x:   An array of dtype `float64`, shape `(n)`, representing the elements in the array x
    :return:    An array of dtype `float64`, shape `(n)`, representing the result of the matrix-vector operation A * x
    """
    # 'U' if the upper-right triangle of A should be used, 'L' if the lower-left triangle should be used
    cdef char* uplo = 'U'
    # The number of rows and columns of the matrix A
    cdef int n = x.shape[0]
    # A scalar to be multiplied with the matrix A
    cdef float64 alpha = 1
    # The increment for the elements of x
    cdef int incx = 1
    # A scalar to be multiplied with vector y
    cdef float64 beta = 0
    # An array of dtype `float64`, shape `(n)`. Will contain the result of A * x
    cdef float64[::1] y = array_float64(n)
    # The increment for the elements of y
    cdef int incy = 1
    # Invoke the DSPMV routine...
    dspmv(uplo, &n, &alpha, &a[0], &x[0], &incx, &beta, &y[0], &incy)
    return y


cdef inline float64[::1] dsysv_float64(float64[::1] coefficients, float64[::1] ordinates,
                                       float64 l2_regularization_weight):
    """
    Computes and returns the solution to a system of linear equations A * X = B using LAPACK's DSYSV solver (see
    http://www.netlib.org/lapack/explore-html/d6/d0e/group__double_s_ysolve_ga9995c47692c9885ed5d6a6b431686f41.html).
    DSYSV requires A to be a double-precision matrix with shape `(num_equations, num_equations)`, representing the
    coefficients, and B to be a double-precision matrix with shape `(num_equations, nrhs)`, representing the ordinates.
    X is a matrix of unknowns with shape `(num_equations, nrhs)`.

    DSYSV will overwrite the matrices A and B. When terminated successfully, B will contain the solution to the system
    of linear equations. To retain their state, this function will copy the given arrays before invoking DSYSV.

    Furthermore, DSYSV assumes the matrix of coefficients A to be symmetrical, i.e., it will only use the upper-right
    triangle of A, whereas the remaining elements are ignored. For reasons of space efficiency, this function expects
    the coefficients to be given as an array with shape `num_equations * (num_equations + 1) // 2`, representing the
    elements of the upper-right triangle of A, where the columns are appended to each other and unspecified elements are
    omitted. This function will implicitly convert the given array into a matrix that is suited for DSYSV.
    
    Optionally, this function allows to specify a weight to be used for L2 regularization. The given weight is added to 
    each element on the diagonal of the matrix of coefficients A.

    :param coefficients:                An array of dtype `float64`, shape `num_equations * (num_equations + 1) // 2)`,
                                        representing coefficients
    :param ordinates:                   An array of dtype `float64`, shape `(num_equations)`, representing the ordinates
    :param l2_regularization_weight:    A scalar of dtype `float64`, representing the weight of the L2 regularization
    :return:                            An array of dtype `float64`, shape `(num_equations)`, representing the solution 
                                        to the system of linear equations
    """
    cdef float64[::1] result
    cdef float64 tmp
    cdef intp r, c, i
    # The number of linear equations
    cdef int n = ordinates.shape[0]
    # Create the array A by copying the array `coefficients`. DSYSV requires the array A to be Fortran-contiguous...
    cdef float64[::1, :] a = matrix_float64(n, n)
    i = 0

    for c in range(n):
        for r in range(c + 1):
            tmp = coefficients[i]

            if r == c:
                tmp += l2_regularization_weight

            a[r, c] = tmp
            i += 1

    # Create the array B by copying the array `ordinates`. It will be overwritten with the solution to the system of
    # linear equations. DSYSV requires the array B to be Fortran-contiguous...
    cdef float64[::1, :] b = matrix_float64(n, 1)

    for r in range(n):
        b[r, 0] = ordinates[r]

    # 'U' if the upper-right triangle of A should be used, 'L' if the lower-left triangle should be used
    cdef char* uplo = 'U'
    # The number of right-hand sides, i.e, the number of columns of the matrix B
    cdef int nrhs = b.shape[1]
    # Variable to hold the result of the solver. Will be 0 when terminated successfully, unlike 0 otherwise
    cdef int info
    # We must query optimal value for the argument `lwork` (the length of the working array `work`)...
    cdef double worksize
    cdef int lwork = -1  # -1 means that the optimal value should be queried
    dsysv(uplo, &n, &nrhs, &a[0, 0], &n, <int*>0, &b[0, 0], &n, &worksize, &lwork, &info)  # Queries the optimal value
    lwork = <int>worksize
    # Allocate the working array...
    cdef double* work = <double*>malloc(lwork * sizeof(double))
    # Allocate another working array...
    cdef int* ipiv = <int*>malloc(n * sizeof(int))

    try:
        # Run the DSYSV solver...
        dsysv(uplo, &n, &nrhs, &a[0, 0], &n, ipiv, &b[0, 0], &n, work, &lwork, &info)

        if info == 0:
            # The solution has been computed successfully...
            result = b[:, 0]
            return result
        else:
            # An error occurred...
            raise ArithmeticError('DSYSV terminated with non-zero info code: ' + str(info))
    finally:
        # Free the allocated memory...
        free(<void*>ipiv)
        free(<void*>work)
