#include "common/multi_threading/multi_threading_manual.hpp"
#include "common/util/threads.hpp"
#include "common/util/validation.hpp"


ManualMultiThreadingConfig::ManualMultiThreadingConfig()
    : numThreads_(0) {

}

uint32 ManualMultiThreadingConfig::getNumThreads() const {
    return numThreads_;
}

IManualMultiThreadingConfig& ManualMultiThreadingConfig::setNumThreads(uint32 numThreads) {
    if (numThreads != 0) { assertGreaterOrEqual<uint32>("numThreads", numThreads, 1); }
    numThreads_ = numThreads;
    return *this;
}

uint32 ManualMultiThreadingConfig::getNumThreads(const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
    return getNumAvailableThreads(numThreads_);
}
