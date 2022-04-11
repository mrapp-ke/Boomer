/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/label_space_info.hpp"
#include "common/input/label_vector.hpp"
#include "common/data/functions.hpp"
#include <unordered_map>
#include <functional>
#include <memory>

/**
 * Defines an interface for all classes that provide access to a set of unique label vectors.
 */
class MLRLCOMMON_API ILabelVectorSet : public ILabelSpaceInfo {

    public:

        virtual ~ILabelVectorSet() override { };

        /**
         * A visitor function for handling objects of the type `LabelVector`.
         */
        typedef std::function<void(const LabelVector&)> LabelVectorVisitor;

        /**
         * Adds a label vector to the set.
         *
         * @param labelVectorPtr An unique pointer to an object of type `LabelVector`
         */
        virtual void addLabelVector(std::unique_ptr<LabelVector> labelVectorPtr) = 0;

        /**
         * Invokes the given visitor function for each label vector that has been added to the set.
         *
         * @param visitor The visitor function for handling objects of the type `LabelVector`
         */
        virtual void visit(LabelVectorVisitor visitor) const = 0;

};

/**
 * An implementation of the type `ILabelVectorSet` that stores a set of unique label vectors, as well as their
 * frequency.
 */
class LabelVectorSet final : public ILabelVectorSet {

    private:

        /**
         * Allows to compute hashes for objects of type `LabelVector`.
         */
        struct Hash {

            inline std::size_t operator()(const std::unique_ptr<LabelVector>& v) const {
                return hashArray(v->cbegin(), v->getNumElements());
            }

        };

        /**
         * Allows to check whether two objects of type `LabelVector` are equal or not.
         */
        struct Pred {

            inline bool operator()(const std::unique_ptr<LabelVector>& lhs,
                                   const std::unique_ptr<LabelVector>& rhs) const {
                return compareArrays(lhs->cbegin(), lhs->getNumElements(), rhs->cbegin(), rhs->getNumElements());
            }

        };

        typedef std::unordered_map<std::unique_ptr<LabelVector>, uint32, Hash, Pred> Map;

        Map labelVectors_;

    public:

        /**
         * An iterator that provides read-only access to the label vectors, as well as their frequency.
         */
        typedef Map::const_iterator const_iterator;

        /**
         * Returns a `const_iterator` to the beginning of the label vectors in the set.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the label vectors in the set.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const;

        void addLabelVector(std::unique_ptr<LabelVector> labelVectorPtr) override;

        void visit(LabelVectorVisitor visitor) const override;

        std::unique_ptr<IClassificationPredictor> createClassificationPredictor(
            const IClassificationPredictorFactory& factory, const RuleList& model) const override;

        std::unique_ptr<IRegressionPredictor> createRegressionPredictor(
            const IRegressionPredictorFactory& factory, const RuleList& model) const override;

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
            const IProbabilityPredictorFactory& factory, const RuleList& model) const override;

};

/**
 * Creates and returns a new object of the type `ILabelVectorSet`.
 *
 * @return An unique pointer to an object of type `ILabelVectorSet` that has been created
 */
MLRLCOMMON_API std::unique_ptr<ILabelVectorSet> createLabelVectorSet();
