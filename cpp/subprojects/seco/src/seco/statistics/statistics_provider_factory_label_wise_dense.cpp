#include "seco/statistics/statistics_provider_factory_label_wise_dense.hpp"
#include "seco/data/matrix_weight_dense.hpp"
#include "seco/data/vector_confusion_matrix_dense.hpp"
#include "common/validation.hpp"
#include "statistics_label_wise_common.hpp"
#include "statistics_provider_label_wise.hpp"


namespace seco {

    /**
     * Provides access to the elements of confusion matrices that are computed independently for each label and are
     * stored using dense data structures.
     */
    template<typename LabelMatrix>
    class DenseLabelWiseStatistics final : public AbstractLabelWiseStatistics<LabelMatrix, DenseWeightMatrix,
                                                                              DenseConfusionMatrixVector,
                                                                              ILabelWiseRuleEvaluationFactory>  {

        public:

            /**
             * @param ruleEvaluationFactory     A reference to an object of type `ILabelWiseRuleEvaluationFactory` that
             *                                  allows to create instances of the class that is used for calculating the
             *                                  predictions, as well as corresponding quality scores, of rules
             * @param labelMatrix               A reference to an object of template type `LabelMatrix` that provides
             *                                  access to the labels of the training examples
             * @param weightMatrixPtr           An unique pointer to an object of type `DenseWeightMatrix` that stores
             *                                  the weights of individual examples and labels
             * @param majorityLabelVectorPtr    An unique pointer to an object of type `BinarySparseArrayVector` that
             *                                  stores the predictions of the default rule
             */
            DenseLabelWiseStatistics(const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory,
                                     const LabelMatrix& labelMatrix, std::unique_ptr<DenseWeightMatrix> weightMatrixPtr,
                                     std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr)
                : AbstractLabelWiseStatistics<LabelMatrix, DenseWeightMatrix, DenseConfusionMatrixVector,
                                              ILabelWiseRuleEvaluationFactory>(
                      ruleEvaluationFactory, labelMatrix, std::move(weightMatrixPtr),
                      std::move(majorityLabelVectorPtr)) {

            }

            void visitWeightMatrix(ICoverageStatistics::DenseWeightMatrixVisitor denseWeightMatrixVisitor) override {
                denseWeightMatrixVisitor(this->weightMatrixPtr_);
            }

    };

    static inline std::unique_ptr<ILabelWiseStatistics<ILabelWiseRuleEvaluationFactory>> createStatistics(
            const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory, const CContiguousLabelMatrix& labelMatrix) {
        uint32 numExamples = labelMatrix.getNumRows();
        uint32 numLabels = labelMatrix.getNumCols();
        std::unique_ptr<DenseWeightMatrix> weightMatrixPtr =
            std::make_unique<DenseWeightMatrix>(numExamples, numLabels);
        std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr =
            std::make_unique<BinarySparseArrayVector>(numLabels);
        BinarySparseArrayVector::index_iterator majorityIterator = majorityLabelVectorPtr->indices_begin();
        float64 threshold = numExamples / 2.0;
        float64 sumOfUncoveredWeights = 0;
        uint32 n = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 numRelevant = 0;

            for (uint32 j = 0; j < numExamples; j++) {
                uint8 trueLabel = labelMatrix.row_values_cbegin(j)[i];
                numRelevant += trueLabel;
            }

            if (numRelevant > threshold) {
                sumOfUncoveredWeights += (numExamples - numRelevant);
                majorityIterator[n] = i;
                n++;
            } else {
                sumOfUncoveredWeights += numRelevant;
            }
        }

        majorityLabelVectorPtr->setNumElements(n, true);
        weightMatrixPtr->setSumOfUncoveredWeights(sumOfUncoveredWeights);
        return std::make_unique<DenseLabelWiseStatistics<CContiguousLabelMatrix>>(ruleEvaluationFactory, labelMatrix,
                                                                                  std::move(weightMatrixPtr),
                                                                                  std::move(majorityLabelVectorPtr));
    }

    static inline std::unique_ptr<ILabelWiseStatistics<ILabelWiseRuleEvaluationFactory>> createStatistics(
            const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory, const CsrLabelMatrix& labelMatrix) {
        uint32 numExamples = labelMatrix.getNumRows();
        uint32 numLabels = labelMatrix.getNumCols();
        std::unique_ptr<DenseWeightMatrix> weightMatrixPtr =
            std::make_unique<DenseWeightMatrix>(numExamples, numLabels);
        std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr =
            std::make_unique<BinarySparseArrayVector>(numLabels, true);
        BinarySparseArrayVector::index_iterator majorityIterator = majorityLabelVectorPtr->indices_begin();

        for (uint32 i = 0; i < numExamples; i++) {
            CsrLabelMatrix::index_const_iterator indexIterator = labelMatrix.row_indices_cbegin(i);
            uint32 numElements = labelMatrix.row_indices_cend(i) - indexIterator;

            for (uint32 j = 0; j < numElements; j++) {
                uint32 index = indexIterator[j];
                majorityIterator[index] += 1;
            }
        }

        float64 threshold = numExamples / 2.0;
        float64 sumOfUncoveredWeights = 0;
        uint32 n = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 numRelevant = majorityIterator[i];

            if (numRelevant > threshold) {
                sumOfUncoveredWeights += (numExamples - numRelevant);
                majorityIterator[n] = i;
                n++;
            } else {
                sumOfUncoveredWeights += numRelevant;
            }
        }

        majorityLabelVectorPtr->setNumElements(n, true);
        weightMatrixPtr->setSumOfUncoveredWeights(sumOfUncoveredWeights);
        return std::make_unique<DenseLabelWiseStatistics<CsrLabelMatrix>>(ruleEvaluationFactory, labelMatrix,
                                                                          std::move(weightMatrixPtr),
                                                                          std::move(majorityLabelVectorPtr));
    }


    DenseLabelWiseStatisticsProviderFactory::DenseLabelWiseStatisticsProviderFactory(
            std::unique_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
            std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
            std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr)
        : defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)) {
        assertNotNull("defaultRuleEvaluationFactoryPtr", defaultRuleEvaluationFactoryPtr_.get());
        assertNotNull("regularRuleEvaluationFactoryPtr", regularRuleEvaluationFactoryPtr_.get());
        assertNotNull("pruningRuleEvaluationFactoryPtr", pruningRuleEvaluationFactoryPtr_.get());
    }

    std::unique_ptr<IStatisticsProvider> DenseLabelWiseStatisticsProviderFactory::create(
            const CContiguousLabelMatrix& labelMatrix) const {
        std::unique_ptr<ILabelWiseStatistics<ILabelWiseRuleEvaluationFactory>> statisticsPtr =
            createStatistics(*defaultRuleEvaluationFactoryPtr_, labelMatrix);
        return std::make_unique<LabelWiseStatisticsProvider<ILabelWiseRuleEvaluationFactory>>(
            *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    std::unique_ptr<IStatisticsProvider> DenseLabelWiseStatisticsProviderFactory::create(
            const CsrLabelMatrix& labelMatrix) const {
        std::unique_ptr<ILabelWiseStatistics<ILabelWiseRuleEvaluationFactory>> statisticsPtr =
            createStatistics(*defaultRuleEvaluationFactoryPtr_, labelMatrix);
        return std::make_unique<LabelWiseStatisticsProvider<ILabelWiseRuleEvaluationFactory>>(
            *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

}
