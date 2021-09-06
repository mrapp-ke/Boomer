/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/view_c_contiguous.hpp"


/**
 * Implements row-wise read-only access to the feature values of individual training examples that are stored in a
 * pre-allocated C-contiguous array.
 */
typedef CContiguousConstView<const float32> CContiguousFeatureMatrix;
