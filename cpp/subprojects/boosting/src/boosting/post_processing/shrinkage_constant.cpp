#include "boosting/post_processing/shrinkage_constant.hpp"

#include "common/util/validation.hpp"

namespace boosting {

    /**
     * Post-processes the predictions of rules by shrinking their weights by a constant shrinkage parameter.
     */
    class ConstantShrinkage final : public IPostProcessor {
        private:

            const float64 shrinkage_;

        public:

            /**
             * @param shrinkage The shrinkage parameter. Must be in (0, 1)
             */
            ConstantShrinkage(float64 shrinkage) : shrinkage_(shrinkage) {}

            /**
             * @see `IPostProcessor::postProcess`
             */
            void postProcess(AbstractPrediction& prediction) const override {
                uint32 numElements = prediction.getNumElements();
                AbstractPrediction::score_iterator iterator = prediction.scores_begin();

                for (uint32 i = 0; i < numElements; i++) {
                    iterator[i] *= shrinkage_;
                }
            }
    };

    /**
     * Allows to create instances of the type `IPostProcessor` that post-process the predictions of rules by shrinking
     * their weights by a constant "shrinkage" parameter.
     */
    class ConstantShrinkageFactory final : public IPostProcessorFactory {
        private:

            const float64 shrinkage_;

        public:

            /**
             * @param shrinkage The value of the "shrinkage" parameter. Must be in (0, 1)
             */
            ConstantShrinkageFactory(float64 shrinkage) : shrinkage_(shrinkage) {}

            /**
             * @see `IPostProcessorFactory::create`
             */
            std::unique_ptr<IPostProcessor> create() const override {
                return std::make_unique<ConstantShrinkage>(shrinkage_);
            }
    };

    ConstantShrinkageConfig::ConstantShrinkageConfig() : shrinkage_(0.3) {}

    float64 ConstantShrinkageConfig::getShrinkage() const {
        return shrinkage_;
    }

    IConstantShrinkageConfig& ConstantShrinkageConfig::setShrinkage(float64 shrinkage) {
        assertGreater<float64>("shrinkage", shrinkage, 0);
        assertLess<float64>("shrinkage", shrinkage, 1);
        shrinkage_ = shrinkage;
        return *this;
    }

    std::unique_ptr<IPostProcessorFactory> ConstantShrinkageConfig::createPostProcessorFactory() const {
        return std::make_unique<ConstantShrinkageFactory>(shrinkage_);
    }

}
