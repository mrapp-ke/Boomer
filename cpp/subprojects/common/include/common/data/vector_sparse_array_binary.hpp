/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/vector_dense.hpp"


/**
 * An one-dimensional sparse vector that stores a fixed number of indices in a C-contiguous array.
 */
typedef DenseVector<uint32> BinarySparseArrayVector;
