#include "boosting/rule_evaluation/rule_evaluation_example_wise_complete_binned.hpp"
#include "boosting/math/math.hpp"
#include "common/rule_evaluation/score_vector_binned_dense.hpp"
#include "common/data/arrays.hpp"
#include "common/validation.hpp"
#include "rule_evaluation_example_wise_complete_common.hpp"
#include "rule_evaluation_label_wise_common.hpp"


namespace boosting {

    template<typename ScoreIterator>
    static inline void calculateLabelWiseScores(
            DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator,
            DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessianIterator,
            ScoreIterator scoreIterator, uint32 numElements, float64 l2RegularizationWeight) {
        for (uint32 i = 0; i < numElements; i++) {
            scoreIterator[i] = calculateLabelWiseScore(gradientIterator[i], hessianIterator[i], l2RegularizationWeight);
        }
    }

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

    template<typename BinIndexIterator>
    static inline void aggregateGradientsAndHessians(
            DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator,
            DenseExampleWiseStatisticVector::hessian_const_iterator hessianIterator, uint32 numElements,
            BinIndexIterator binIndexIterator, const uint32* binIndices, float64* gradients, float64* hessians,
            uint32 maxBins) {
        for (uint32 i = 0; i < numElements; i++) {
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

        for (uint32 i = 1; i < numElements; i++) {
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

    static inline void addRegularizationWeight(float64* coefficients, uint32 numPredictions, const uint32* weights,
                                               float64 l2RegularizationWeight) {
        for (uint32 i = 0; i < numPredictions; i++) {
            uint32 weight = weights[i];
            coefficients[(i * numPredictions) + i] += (weight * l2RegularizationWeight);
        }
    }

    /**
     * Allows to calculate the predictions of complete rules, as well as an overall quality score, based on the
     * gradients and Hessians that are stored by a `DenseExampleWiseStatisticVector` using L2 regularization. The labels
     * are assigned to bins based on the gradients and Hessians.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class DenseExampleWiseCompleteBinnedRuleEvaluation final :
            public AbstractExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector, T> {

        private:

            uint32 maxBins_;

            DenseBinnedScoreVector<T> scoreVector_;

            float64* aggregatedGradients_;

            float64* aggregatedHessians_;

            uint32* binIndices_;

            uint32* numElementsPerBin_;

            float64* criteria_;

            float64 l2RegularizationWeight_;

            std::unique_ptr<ILabelBinning> binningPtr_;

            const Blas& blas_;

            const Lapack& lapack_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param maxBins                   The maximum number of bins
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             * @param blas                      A reference to an object of type `Blas` that allows to execute different
             *                                  BLAS routines
             * @param lapack                    A reference to an object of type `Lapack` that allows to execute
             *                                  different LAPACK routines
             */
            DenseExampleWiseCompleteBinnedRuleEvaluation(const T& labelIndices, uint32 maxBins,
                                                         float64 l2RegularizationWeight,
                                                         std::unique_ptr<ILabelBinning> binningPtr, const Blas& blas,
                                                         const Lapack& lapack)
                : AbstractExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector, T>(maxBins, lapack),
                  maxBins_(maxBins), scoreVector_(DenseBinnedScoreVector<T>(labelIndices, maxBins + 1)),
                  aggregatedGradients_(new float64[maxBins]),
                  aggregatedHessians_(new float64[triangularNumber(maxBins)]), binIndices_(new uint32[maxBins]),
                  numElementsPerBin_(new uint32[maxBins]), criteria_(new float64[labelIndices.getNumElements()]),
                  l2RegularizationWeight_(l2RegularizationWeight), binningPtr_(std::move(binningPtr)), blas_(blas),
                  lapack_(lapack) {
                // The last bin is used for labels for which the corresponding criterion is zero. For this particular
                // bin, the prediction is always zero.
                scoreVector_.scores_binned_begin()[maxBins_] = 0;
            }

            ~DenseExampleWiseCompleteBinnedRuleEvaluation() {
                delete[] aggregatedGradients_;
                delete[] aggregatedHessians_;
                delete[] binIndices_;
                delete[] numElementsPerBin_;
                delete[] criteria_;
            }

            const IScoreVector& calculatePrediction(DenseExampleWiseStatisticVector& statisticVector) override {
                // Calculate label-wise criteria...
                uint32 numLabels = statisticVector.getNumElements();
                DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator =
                    statisticVector.gradients_cbegin();
                DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessianIterator =
                    statisticVector.hessians_diagonal_cbegin();
                calculateLabelWiseScores(gradientIterator, hessianIterator, criteria_, numLabels,
                                         l2RegularizationWeight_);

                // Obtain information about the bins to be used...
                LabelInfo labelInfo = binningPtr_->getLabelInfo(criteria_, numLabels);
                uint32 numBins = labelInfo.numPositiveBins + labelInfo.numNegativeBins;

                if (numBins > 0) {
                    // Reset arrays to zero...
                    setArrayToZeros(numElementsPerBin_, numBins);

                    // Apply binning method in order to aggregate the gradients and Hessians that belong to the same
                    // bins...
                    typename DenseBinnedScoreVector<T>::index_binned_iterator binIndexIterator =
                        scoreVector_.indices_binned_begin();
                    auto callback = [=](uint32 binIndex, uint32 labelIndex) {
                        numElementsPerBin_[binIndex] += 1;
                        binIndexIterator[labelIndex] = binIndex;
                    };
                    auto zeroCallback = [=](uint32 labelIndex) {
                        binIndexIterator[labelIndex] = maxBins_;
                    };
                    binningPtr_->createBins(labelInfo, criteria_, numLabels, callback, zeroCallback);

                    // Determine number of non-empty bins...
                    numBins = removeEmptyBins(numElementsPerBin_, binIndices_, numBins);
                    scoreVector_.setNumBins(numBins, false);

                    // Aggregate gradients and Hessians...
                    setArrayToZeros(aggregatedGradients_, numBins);
                    setArrayToZeros(aggregatedHessians_, triangularNumber(numBins));
                    aggregateGradientsAndHessians(gradientIterator, statisticVector.hessians_cbegin(), numLabels,
                                                  binIndexIterator, binIndices_, aggregatedGradients_,
                                                  aggregatedHessians_, maxBins_);

                    // Copy Hessians to the matrix of coefficients and add regularization weight to its diagonal...
                    copyCoefficients(aggregatedHessians_, this->dsysvTmpArray1_, numBins);
                    addRegularizationWeight(this->dsysvTmpArray1_, numBins, numElementsPerBin_,
                                            l2RegularizationWeight_);

                    // Copy gradients to the vector of ordinates...
                    typename DenseBinnedScoreVector<T>::score_binned_iterator scoreIterator =
                        scoreVector_.scores_binned_begin();
                    copyOrdinates(aggregatedGradients_, scoreIterator, numBins);

                    // Calculate the scores to be predicted for the individual labels by solving a system of linear
                    // equations...
                    lapack_.dsysv(this->dsysvTmpArray1_, this->dsysvTmpArray2_, this->dsysvTmpArray3_, scoreIterator,
                                  numBins, this->dsysvLwork_);

                    // Calculate the overall quality score...
                    float64 qualityScore = calculateOverallQualityScore(scoreIterator, aggregatedGradients_,
                                                                        aggregatedHessians_, this->dspmvTmpArray_,
                                                                        numBins, blas_);

                    // Evaluate regularization term...
                    float64 regularizationTerm = 0.5 * l2RegularizationWeight_
                                                 * l2NormPow(scoreIterator, numElementsPerBin_, numBins);

                    scoreVector_.overallQualityScore = qualityScore + regularizationTerm;
                } else {
                    setArrayToValue(scoreVector_.indices_binned_begin(), numLabels, maxBins_);
                    scoreVector_.overallQualityScore = 0;
                }

                return scoreVector_;
            }

    };

    ExampleWiseCompleteBinnedRuleEvaluationFactory::ExampleWiseCompleteBinnedRuleEvaluationFactory(
            float64 l2RegularizationWeight, std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr,
            std::unique_ptr<Blas> blasPtr, std::unique_ptr<Lapack> lapackPtr)
        : l2RegularizationWeight_(l2RegularizationWeight), labelBinningFactoryPtr_(std::move(labelBinningFactoryPtr)),
          blasPtr_(std::move(blasPtr)), lapackPtr_(std::move(lapackPtr)) {
        assertGreaterOrEqual<float64>("l2RegularizationWeight", l2RegularizationWeight, 0);
        assertNotNull("labelBinningFactoryPtr", labelBinningFactoryPtr_.get());
        assertNotNull("blasPtr", blasPtr_.get());
        assertNotNull("lapackPtr", lapackPtr_.get());
    }

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> ExampleWiseCompleteBinnedRuleEvaluationFactory::create(
            const DenseExampleWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        uint32 maxBins = labelBinningPtr->getMaxBins(indexVector.getNumElements());
        return std::make_unique<DenseExampleWiseCompleteBinnedRuleEvaluation<CompleteIndexVector>>(
            indexVector, maxBins, l2RegularizationWeight_, std::move(labelBinningPtr), *blasPtr_, *lapackPtr_);
    }

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> ExampleWiseCompleteBinnedRuleEvaluationFactory::create(
            const DenseExampleWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        uint32 maxBins = labelBinningPtr->getMaxBins(indexVector.getNumElements());
        return std::make_unique<DenseExampleWiseCompleteBinnedRuleEvaluation<PartialIndexVector>>(
            indexVector, maxBins, l2RegularizationWeight_, std::move(labelBinningPtr), *blasPtr_, *lapackPtr_);
    }

}
