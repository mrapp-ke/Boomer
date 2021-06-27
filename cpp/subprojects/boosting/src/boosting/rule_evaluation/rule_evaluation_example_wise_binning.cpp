#include "boosting/rule_evaluation/rule_evaluation_example_wise_binning.hpp"
#include "boosting/binning/label_binning_equal_width.hpp"
#include "boosting/math/blas.hpp"
#include "common/data/arrays.hpp"
#include "common/rule_evaluation/score_vector_label_wise_binned_dense.hpp"
#include "rule_evaluation_label_wise_binning_common.hpp"
#include "rule_evaluation_example_wise_common.hpp"
#include <cstdlib>


namespace boosting {

    /**
     * The type of the class that allows to assign labels to bins, based on the gradients and Hessians that are stored
     * in a `DenseExampleWiseStatisticVector`.
     */
    typedef ILabelBinning<DenseExampleWiseStatisticVector::gradient_const_iterator,
                          DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator> LabelBinningType;

    static inline uint32 removeEmptyBins(uint32* numElementsPerBin, uint32* binIndices, uint32 numBins) {
        uint32 n = 0;

        for (uint32 i = 0; i < numBins; i++) {
            binIndices[i] = n;
            uint32 numElements = numElementsPerBin[i];

            if (numElements > 0) {
                numElementsPerBin[n] = numElements;
                n++;
            }
        }

        return n;
    }

    template<class T>
    static inline void aggregateGradientsAndHessians(const DenseExampleWiseStatisticVector& statisticVector,
                                                     const uint32* binIndices, DenseBinnedScoreVector<T>& scoreVector,
                                                     float64* gradients, float64* hessians, uint32 maxBins) {
        uint32 numLabels = statisticVector.getNumElements();
        DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator = statisticVector.gradients_cbegin();
        DenseExampleWiseStatisticVector::hessian_const_iterator hessianIterator = statisticVector.hessians_cbegin();
        typename DenseBinnedScoreVector<T>::index_binned_iterator binIndexIterator = scoreVector.indices_binned_begin();

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 originalBinIndex = binIndexIterator[i];

            if (originalBinIndex != maxBins) {
                uint32 binIndex = binIndices[originalBinIndex];
                binIndexIterator[i] = binIndex;

                // Add the gradient that corresponds to the `i`-th element of the original gradient vector to the
                // corresponding element of the aggregated gradient vector...
                gradients[binIndex] += gradientIterator[i];

                // Add the Hessian that corresponds to the `i`-th element on the diagonal of the original Hessian matrix to
                // the corresponding element of the aggregated Hessian matrix...
                hessians[triangularNumber(binIndex + 1) - 1] += hessianIterator[triangularNumber(i + 1) - 1];
            }
        }

        for (uint32 i = 1; i < numLabels; i++) {
            uint32 binIndex = binIndexIterator[i];

            if (binIndex != maxBins) {
                for (uint32 j = 0; j < i; j++) {
                    uint32 binIndex2 = binIndexIterator[j];

                    // Add the hessian at the `i`-th row and `j`-th column of the original Hessian matrix to the
                    // corresponding element of the aggregated Hessian matrix, if the labels at indices `i` and `j` do not
                    // belong to the same bin...
                    if (binIndex2 != maxBins && binIndex != binIndex2) {
                        uint32 r, c;

                        if (binIndex < binIndex2) {
                            r = binIndex;
                            c = binIndex2;
                        } else {
                            r = binIndex2;
                            c = binIndex;
                        }

                        hessians[triangularNumber(c) + r] += hessianIterator[triangularNumber(i) + j];
                    }
                }
            }
        }
    }

    /**
     * Adds a specific L2 regularization weight to the diagonal of a coefficient matrix.
     *
     * @param output                    A pointer to an array of type `float64`, shape `(n, n)` that stores the
     *                                  coefficients
     * @param n                         The number of rows and columns in the coefficient matrix
     * @param numElementsPerBin         A pointer to an array of type `uint32`, shape `(n)` that stores the number of
     *                                  elements per bin
     * @param l2RegularizationWeight    The L2 regularization weight to be added
     */
    static inline void addRegularizationWeight(float64* output, uint32 n, const uint32* numElementsPerBin,
                                               float64 l2RegularizationWeight) {
        for (uint32 i = 0; i < n; i++) {
            uint32 weight = numElementsPerBin[i];
            output[(i * n) + i] += (weight * l2RegularizationWeight);
        }
    }

    /**
     * Allows to calculate the predictions of rules, as well as corresponding quality scores, based on the gradients and
     * Hessians that have been calculated according to a loss function that is applied example wise using L2
     * regularization. The labels are assigned to bins based on the corresponding gradients.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<class T>
    class BinningExampleWiseRuleEvaluation : public AbstractExampleWiseRuleEvaluation<T> {

        private:

            float64 l2RegularizationWeight_;

            uint32 maxBins_;

            std::unique_ptr<LabelBinningType> binningPtr_;

            std::shared_ptr<Blas> blasPtr_;

            DenseBinnedScoreVector<T>* scoreVector_;

            DenseBinnedLabelWiseScoreVector<T>* labelWiseScoreVector_;

            float64* tmpGradients_;

            float64* tmpHessians_;

            uint32* numElementsPerBin_;

            uint32* binIndices_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param maxBins                   The maximum number of bins to assign labels to
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             * @param blasPtr                   A shared pointer to an object of type `Blas` that allows to execute
             *                                  different BLAS routines
             * @param lapackPtr                 A shared pointer to an object of type `Lapack` that allows to execute
             *                                  different LAPACK routines
             */
            BinningExampleWiseRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight, uint32 maxBins,
                                             std::unique_ptr<LabelBinningType> binningPtr,
                                             std::shared_ptr<Blas> blasPtr, std::shared_ptr<Lapack> lapackPtr)
                : AbstractExampleWiseRuleEvaluation<T>(labelIndices, lapackPtr),
                  l2RegularizationWeight_(l2RegularizationWeight), maxBins_(maxBins),
                  binningPtr_(std::move(binningPtr)), blasPtr_(blasPtr), scoreVector_(nullptr),
                  labelWiseScoreVector_(nullptr), tmpGradients_(nullptr), tmpHessians_(nullptr),
                  numElementsPerBin_(nullptr), binIndices_(nullptr) {

            }

            ~BinningExampleWiseRuleEvaluation() {
                delete scoreVector_;
                delete labelWiseScoreVector_;
                free(tmpGradients_);
                free(tmpHessians_);
                free(numElementsPerBin_);
                free(binIndices_);
            }

            const ILabelWiseScoreVector& calculateLabelWisePrediction(
                    const DenseExampleWiseStatisticVector& statisticVector) override {
                if (labelWiseScoreVector_ == nullptr) {
                    labelWiseScoreVector_ = new DenseBinnedLabelWiseScoreVector<T>(this->labelIndices_, maxBins_ + 1);
                    tmpGradients_ = (float64*) malloc(maxBins_ * sizeof(float64));
                    tmpHessians_ = (float64*) malloc(maxBins_ * sizeof(float64));
                    numElementsPerBin_ = (uint32*) malloc(maxBins_ * sizeof(uint32));

                    // The last bin is used for labels with zero statistics. For this particular bin, the prediction and
                    // quality score is always zero.
                    labelWiseScoreVector_->scores_binned_begin()[maxBins_] = 0;
                    labelWiseScoreVector_->quality_scores_binned_begin()[maxBins_] = 0;
                }

                // Obtain information about the bins to be used...
                LabelInfo labelInfo = binningPtr_->getLabelInfo(statisticVector.gradients_cbegin(),
                                                                statisticVector.gradients_cend(),
                                                                statisticVector.hessians_diagonal_cbegin(),
                                                                statisticVector.hessians_diagonal_cend(),
                                                                l2RegularizationWeight_);
                uint32 numBins = labelInfo.numPositiveBins + labelInfo.numNegativeBins;
                labelWiseScoreVector_->setNumBins(numBins, false);

                // Reset arrays to zero...
                setArrayToZeros(tmpGradients_, numBins);
                setArrayToZeros(tmpHessians_, numBins);
                setArrayToZeros(numElementsPerBin_, numBins);

                // Apply binning method in order to aggregate the gradients and Hessians that belong to the same bins...
                auto callback = [this, &statisticVector](uint32 binIndex, uint32 labelIndex, float64 statistic) {
                    tmpGradients_[binIndex] += statisticVector.gradients_cbegin()[labelIndex];
                    tmpHessians_[binIndex] += statisticVector.hessians_diagonal_cbegin()[labelIndex];
                    numElementsPerBin_[binIndex] += 1;
                    labelWiseScoreVector_->indices_binned_begin()[labelIndex] = binIndex;
                };
                auto zeroCallback = [this](uint32 labelIndex) {
                    labelWiseScoreVector_->indices_binned_begin()[labelIndex] = maxBins_;
                };
                binningPtr_->createBins(labelInfo, statisticVector.gradients_cbegin(), statisticVector.gradients_cend(),
                                        statisticVector.hessians_diagonal_cbegin(),
                                        statisticVector.hessians_diagonal_cend(), l2RegularizationWeight_, callback,
                                        zeroCallback);

                // Compute predictions and quality scores...
                labelWiseScoreVector_->overallQualityScore = calculateLabelWisePredictionInternally<
                        typename DenseBinnedLabelWiseScoreVector<T>::score_binned_iterator,
                        typename DenseBinnedLabelWiseScoreVector<T>::quality_score_binned_iterator, float64*, float64*,
                        uint32*>(
                    numBins, labelWiseScoreVector_->scores_binned_begin(),
                    labelWiseScoreVector_->quality_scores_binned_begin(), tmpGradients_, tmpHessians_,
                    numElementsPerBin_, l2RegularizationWeight_);
                return *labelWiseScoreVector_;
            }

            const IScoreVector& calculateExampleWisePrediction(
                    DenseExampleWiseStatisticVector& statisticVector) override {
                if (scoreVector_ == nullptr) {
                    scoreVector_ = new DenseBinnedScoreVector<T>(this->labelIndices_, maxBins_ + 1);
                    this->initializeTmpArrays(maxBins_);
                    tmpGradients_ = (float64*) malloc(maxBins_ * sizeof(float64));
                    tmpHessians_ = (float64*) malloc(triangularNumber(maxBins_) * sizeof(float64));
                    numElementsPerBin_ = (uint32*) malloc(maxBins_ * sizeof(uint32));
                    binIndices_ = (uint32*) malloc(maxBins_ * sizeof(uint32));

                    // The last bin is used for labels with zero statistics. For this particular bin, the prediction is
                    // always zero.
                    scoreVector_->scores_binned_begin()[maxBins_] = 0;
                }

                // Obtain information about the bins to be used...
                LabelInfo labelInfo = binningPtr_->getLabelInfo(statisticVector.gradients_cbegin(),
                                                                statisticVector.gradients_cend(),
                                                                statisticVector.hessians_diagonal_cbegin(),
                                                                statisticVector.hessians_diagonal_cend(),
                                                                l2RegularizationWeight_);
                uint32 numBins = labelInfo.numPositiveBins + labelInfo.numNegativeBins;
                float64 qualityScore;

                if (numBins > 0) {
                    // Reset arrays to zero...
                    setArrayToZeros(numElementsPerBin_, numBins);

                    // Apply binning method in order to aggregate the gradients and Hessians that belong to the same
                    // bins...
                    auto callback = [this](uint32 binIndex, uint32 labelIndex, float64 statistic) {
                        numElementsPerBin_[binIndex] += 1;
                        scoreVector_->indices_binned_begin()[labelIndex] = binIndex;
                    };
                    auto zeroCallback = [this](uint32 labelIndex) {
                        scoreVector_->indices_binned_begin()[labelIndex] = maxBins_;
                    };
                    binningPtr_->createBins(labelInfo, statisticVector.gradients_cbegin(),
                                            statisticVector.gradients_cend(),
                                            statisticVector.hessians_diagonal_cbegin(),
                                            statisticVector.hessians_diagonal_cend(), l2RegularizationWeight_, callback,
                                            zeroCallback);
                    // Determine number of non-empty bins...
                    numBins = removeEmptyBins(numElementsPerBin_, binIndices_, numBins);
                    scoreVector_->setNumBins(numBins, false);

                    // Aggregate gradients and Hessians...
                    setArrayToZeros(tmpGradients_, numBins);
                    setArrayToZeros(tmpHessians_, triangularNumber(numBins));
                    aggregateGradientsAndHessians<T>(statisticVector, binIndices_, *scoreVector_, tmpGradients_,
                                                     tmpHessians_, maxBins_);

                    typename DenseBinnedScoreVector<T>::score_binned_iterator scoreIterator =
                        scoreVector_->scores_binned_begin();
                    copyCoefficients<float64*>(tmpHessians_, this->dsysvTmpArray1_, numBins);
                    addRegularizationWeight(this->dsysvTmpArray1_, numBins, numElementsPerBin_,
                                            l2RegularizationWeight_);
                    copyOrdinates<float64*>(tmpGradients_, scoreIterator, numBins);

                    // Calculate the scores to be predicted for the individual labels by solving a system of linear
                    // equations...
                    this->lapackPtr_->dsysv(this->dsysvTmpArray1_, this->dsysvTmpArray2_, this->dsysvTmpArray3_,
                                            scoreIterator, numBins, this->dsysvLwork_);

                    // Calculate the overall quality score...
                    qualityScore = calculateExampleWiseQualityScore(numBins, scoreIterator, tmpGradients_, tmpHessians_,
                                                                    *blasPtr_, this->dspmvTmpArray_);
                    qualityScore += 0.5 * l2RegularizationWeight_ *
                                    l2NormPow<typename DenseBinnedScoreVector<T>::score_binned_iterator, uint32*>(
                                        scoreIterator, numElementsPerBin_, numBins);
                } else {
                    setArrayToValue(scoreVector_->indices_binned_begin(), this->labelIndices_.getNumElements(),
                                    maxBins_);
                    qualityScore = 0;
                }

                scoreVector_->overallQualityScore = qualityScore;
                return *scoreVector_;
            }

    };

    EqualWidthBinningExampleWiseRuleEvaluationFactory::EqualWidthBinningExampleWiseRuleEvaluationFactory(
            float64 l2RegularizationWeight, float32 binRatio, uint32 minBins, uint32 maxBins,
            std::shared_ptr<Blas> blasPtr, std::shared_ptr<Lapack> lapackPtr)
        : l2RegularizationWeight_(l2RegularizationWeight), binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins),
          blasPtr_(blasPtr), lapackPtr_(lapackPtr) {

    }

    std::unique_ptr<IExampleWiseRuleEvaluation> EqualWidthBinningExampleWiseRuleEvaluationFactory::create(
            const FullIndexVector& indexVector) const {
        std::unique_ptr<LabelBinningType> binningPtr =
            std::make_unique<EqualWidthLabelBinning<DenseExampleWiseStatisticVector::gradient_const_iterator,
                                                    DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator>>(
                                                        binRatio_, minBins_, maxBins_);
        uint32 maxBins = binningPtr->getMaxBins(indexVector.getNumElements());
        return std::make_unique<BinningExampleWiseRuleEvaluation<FullIndexVector>>(indexVector, l2RegularizationWeight_,
                                                                                   maxBins, std::move(binningPtr),
                                                                                   blasPtr_, lapackPtr_);
    }

    std::unique_ptr<IExampleWiseRuleEvaluation> EqualWidthBinningExampleWiseRuleEvaluationFactory::create(
            const PartialIndexVector& indexVector) const {
        std::unique_ptr<LabelBinningType> binningPtr =
            std::make_unique<EqualWidthLabelBinning<DenseExampleWiseStatisticVector::gradient_const_iterator,
                                                    DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator>>(
                                                        binRatio_, minBins_, maxBins_);
        uint32 maxBins = binningPtr->getMaxBins(indexVector.getNumElements());
        return std::make_unique<BinningExampleWiseRuleEvaluation<PartialIndexVector>>(indexVector,
                                                                                      l2RegularizationWeight_, maxBins,
                                                                                      std::move(binningPtr), blasPtr_,
                                                                                      lapackPtr_);
    }

}
