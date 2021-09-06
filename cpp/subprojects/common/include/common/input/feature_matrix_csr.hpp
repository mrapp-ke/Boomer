/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/view_csr.hpp"


/**
 * Implements row-wise read-only access to the feature values of individual training examples that are stored in a
 * pre-allocated sparse matrix in the compressed sparse row (CSR) format.
 */
typedef CsrConstView<const float32> CsrFeatureMatrix;
