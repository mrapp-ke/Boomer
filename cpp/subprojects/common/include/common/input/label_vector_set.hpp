/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/label_vector.hpp"
#include "common/data/functions.hpp"
#include <unordered_map>
#include <functional>
#include <memory>


/**
 * A set that stores unique label vectors, as well as their frequency.
 */
class LabelVectorSet final {

    private:

        /**
         * Allows to compute hashes for objects of type `LabelVector`.
         */
        struct Hash {

            inline std::size_t operator()(const std::unique_ptr<LabelVector>& v) const {
                return hashArray(v->indices_cbegin(), v->getNumElements());
            }

        };

        /**
         * Allows to check whether two objects of type `LabelVector` are equal or not.
         */
        struct Pred {

            inline bool operator()(const std::unique_ptr<LabelVector>& lhs,
                                   const std::unique_ptr<LabelVector>& rhs) const {
                return compareArrays(lhs->indices_cbegin(), lhs->getNumElements(), rhs->indices_cbegin(),
                                     rhs->getNumElements());
            }

        };

        typedef std::unordered_map<std::unique_ptr<LabelVector>, uint32, Hash, Pred> Map;

        Map labelVectors_;

    public:

        /**
         * A visitor function for handling objects of the type `LabelVector`.
         */
        typedef std::function<void(const LabelVector&)> LabelVectorVisitor;

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

        /**
         * Adds a label vector to the set.
         *
         * @param labelVectorPtr An unique pointer to an object of type `LabelVector`
         */
        void addLabelVector(std::unique_ptr<LabelVector> labelVectorPtr);

        /**
         * Invokes the given visitor function for each label vector that has been added to the set.
         *
         * @param visitor The visitor function for handling objects of the type `LabelVector`
         */
        void visit(LabelVectorVisitor visitor) const;

};
