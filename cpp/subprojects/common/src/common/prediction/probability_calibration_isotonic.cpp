#include "common/prediction/probability_calibration_isotonic.hpp"

#include "common/math/math.hpp"

static inline void sortByThresholdsAndEliminateDuplicates(ListOfLists<Tuple<float64>>::row bins) {
    // Sort bins in increasing order by their threshold...
    std::sort(bins.begin(), bins.end(), [=](const Tuple<float64>& lhs, const Tuple<float64>& rhs) {
        return lhs.first < rhs.first;
    });

    // Aggregate adjacent bins with identical thresholds by averaging their probabilities...
    uint32 numBins = (uint32) bins.size();
    uint32 previousIndex = 0;
    Tuple<float64> previousBin = bins[previousIndex];
    uint32 n = 0;

    for (uint32 j = 1; j < numBins; j++) {
        const Tuple<float64>& currentBin = bins[j];

        if (isEqual(currentBin.first, previousBin.first)) {
            uint32 numAggregated = j - previousIndex + 1;
            previousBin.second = iterativeArithmeticMean(numAggregated, currentBin.second, previousBin.second);
        } else {
            bins[n] = previousBin;
            n++;
            previousIndex = j;
            previousBin = currentBin;
        }
    }

    bins[n] = bins[numBins - 1];
    n++;
    bins.resize(n);
}

static inline void aggregateNonIncreasingBins(ListOfLists<Tuple<float64>>::row bins) {
    // We apply the "pool adjacent violators algorithm" (PAVA) to merge adjacent bins with non-increasing
    // probabilities. A temporary array `pools` is used to mark the beginning and end of subsequences with
    // non-increasing probabilities. If such a subsequence was found in range [i, j] then `pools[i] = j` and
    // `pools[j] = i`...
    uint32 numBins = (uint32) bins.size();
    uint32* pools = new uint32[numBins];
    setArrayToIncreasingValues<uint32>(pools, numBins, 0, 1);
    uint32 i = 0;
    uint32 j = 0;

    while (i < numBins && j < numBins && (j = pools[i] + 1) < numBins) {
        Tuple<float64>& previousBin = bins[i];
        Tuple<float64>& currentBin = bins[j];

        // Check if the probabilities of the adjacent bins are monotonically increasing...
        if (currentBin.second > previousBin.second) {
            // The probabilities are increasing, i.e., the monotonicity constraint is not violated, and we can
            // continue with the subsequent bins...
            i = j;
        } else {
            // The probabilities are not increasing, i.e., the monotonicity constraint is violated, and we have to
            // average the probabilities of all bins within the non-increasing subsequence...
            uint32 numBinsInSubsequence = 2;
            previousBin.second = iterativeArithmeticMean(numBinsInSubsequence, currentBin.second, previousBin.second);

            // Search for the end of the non-increasing subsequence...
            while ((j = pools[j] + 1) < numBins) {
                Tuple<float64>& nextBin = bins[j];

                if (nextBin.second > currentBin.second) {
                    // We reached the end of the non-increasing subsequence...
                    break;
                } else {
                    // We are still within the non-increasing subsequence...
                    numBinsInSubsequence++;
                    previousBin.second =
                      iterativeArithmeticMean(numBinsInSubsequence, nextBin.second, previousBin.second);
                    currentBin = nextBin;
                }
            }

            // Store the beginning and end of the current subsequence...
            pools[i] = j - 1;
            pools[j - 1] = i;

            // Restart at the previous subsequence if there is one...
            if (i > 0) {
                j = pools[i - 1];
                i = j;
            }
        }
    }

    // Only keep the first bin within each subsequence...
    j = 0;

    for (i = 0; i < numBins; i = pools[i] + 1) {
        bins[j] = bins[i];
        j++;
    }

    delete[] pools;
    bins.resize(j);
    bins.shrink_to_fit();
}

static inline float64 calibrateProbability(ListOfLists<Tuple<float64>>::const_row bins, float64 probability) {
    // Find the bins that impose a lower and upper bound on the probability...
    ListOfLists<Tuple<float64>>::const_iterator begin = bins.cbegin();
    ListOfLists<Tuple<float64>>::const_iterator end = bins.cend();
    ListOfLists<Tuple<float64>>::const_iterator it =
      std::lower_bound(begin, end, probability, [=](const Tuple<float64>& lhs, const float64& rhs) {
          return lhs.first < rhs;
      });
    uint32 offset = it - begin;
    Tuple<float64> lowerBound;
    Tuple<float64> upperBound;

    if (it == end) {
        lowerBound = begin[offset - 1];
        upperBound = 1;
    } else {
        if (offset > 0) {
            lowerBound = begin[offset - 1];
        } else {
            lowerBound = 0;
        }

        upperBound = *it;
    }

    // Interpolate linearly between the probabilities associated with the lower and upper bound...
    float64 t = (probability - lowerBound.first) / (upperBound.first - lowerBound.first);
    return lowerBound.second + (t * (upperBound.second - lowerBound.second));
}

IsotonicProbabilityCalibrationModel::IsotonicProbabilityCalibrationModel(uint32 numLists)
    : binsPerList_(ListOfLists<Tuple<float64>>(numLists)) {}

IsotonicProbabilityCalibrationModel::bin_list IsotonicProbabilityCalibrationModel::operator[](uint32 listIndex) {
    return binsPerList_[listIndex];
}

IsotonicProbabilityCalibrationModel::const_bin_list IsotonicProbabilityCalibrationModel::operator[](
  uint32 listIndex) const {
    return binsPerList_[listIndex];
}

void IsotonicProbabilityCalibrationModel::fit() {
    uint32 numLists = binsPerList_.getNumRows();

    for (uint32 i = 0; i < numLists; i++) {
        ListOfLists<Tuple<float64>>::row bins = binsPerList_[i];
        sortByThresholdsAndEliminateDuplicates(bins);
        aggregateNonIncreasingBins(bins);
    }
}

float64 IsotonicProbabilityCalibrationModel::calibrateMarginalProbability(uint32 labelIndex,
                                                                          float64 marginalProbability) const {
    return calibrateProbability(binsPerList_[labelIndex], marginalProbability);
}

float64 IsotonicProbabilityCalibrationModel::calibrateJointProbability(uint32 labelVectorIndex,
                                                                       float64 jointProbability) const {
    return calibrateProbability(binsPerList_[labelVectorIndex], jointProbability);
}

uint32 IsotonicProbabilityCalibrationModel::getNumBinLists() const {
    return binsPerList_.getNumRows();
}

void IsotonicProbabilityCalibrationModel::addBin(uint32 listIndex, float64 threshold, float64 probability) {
    ListOfLists<Tuple<float64>>::row row = binsPerList_[listIndex];
    row.emplace_back(threshold, probability);
}

void IsotonicProbabilityCalibrationModel::visit(BinVisitor visitor) const {
    uint32 numLists = binsPerList_.getNumRows();

    for (uint32 i = 0; i < numLists; i++) {
        ListOfLists<Tuple<float64>>::const_row bins = binsPerList_[i];

        for (auto it = bins.cbegin(); it != bins.cend(); it++) {
            const Tuple<float64>& bin = *it;
            visitor(i, bin.first, bin.second);
        }
    }
}

std::unique_ptr<IIsotonicProbabilityCalibrationModel> createIsotonicProbabilityCalibrationModel(uint32 numLists) {
    return std::make_unique<IsotonicProbabilityCalibrationModel>(numLists);
}
