#include "common/multi_threading/multi_threading_no.hpp"


uint32 NoMultiThreadingConfig::getNumThreads(const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
    return 1;
}
