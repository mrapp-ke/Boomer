/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/input/feature_matrix_c_contiguous.hpp"
#include "common/input/feature_matrix_csr.hpp"
#include <functional>

// Forward declarations
class EmptyBody;
class ConjunctiveBody;


/**
 * Defines an interface for all classes that represent the body of a rule.
 */
class IBody {

    public:

        virtual ~IBody() { };

        /**
         * A visitor function for handling objects of the type `EmptyBody`.
         */
        typedef std::function<void(const EmptyBody&)> EmptyBodyVisitor;

        /**
         * A visitor function for handling objects of the type `ConjunctiveBody`.
         */
        typedef std::function<void(const ConjunctiveBody&)> ConjunctiveBodyVisitor;

        /**
         * Returns whether an individual example, which is stored in a C-contiguous matrix, is covered by the body or
         * not.
         *
         * @param begin An iterator to the beginning of the example's feature values
         * @param end   An iterator to the end of the example's feature values
         * @return      True, if the example is covered, false otherwise
         */
        virtual bool covers(CContiguousFeatureMatrix::const_iterator begin,
                            CContiguousFeatureMatrix::const_iterator end) const = 0;

        /**
         * Returns whether an individual example, which is stored in a CSR sparse matrix, is covered by the body or not.
         *
         * @param indicesBegin  An iterator to the beginning of the example's feature values
         * @param indicesEnd    An iterator to the end of the example's feature values
         * @param valuesBegin   An iterator to the beginning of the example's feature_indices
         * @param valuesEnd     An iterator to the end of the example's feature indices
         * @param tmpArray1     An array of type `float32`, shape `(num_features)` that is used to temporarily store
         *                      non-zero feature values. May contain arbitrary values
         * @param tmpArray2     An array of type `uint32`, shape `(num_features)` that is used to temporarily keep track
         *                      of the feature indices with non-zero feature values. Must not contain any elements with
         *                      value `n`
         * @param n             An arbitrary number. If this function is called multiple times for different examples,
         *                      but using the same `tmpArray2`, the number must be unique for each of the function
         *                      invocations
         * @return              True, if the example is covered, false otherwise
         */
        virtual bool covers(CsrFeatureMatrix::index_const_iterator indicesBegin,
                            CsrFeatureMatrix::index_const_iterator indicesEnd,
                            CsrFeatureMatrix::value_const_iterator valuesBegin,
                            CsrFeatureMatrix::value_const_iterator valuesEnd, float32* tmpArray1, uint32* tmpArray2,
                            uint32 n) const = 0;

        /**
         * Invokes one of the given visitor functions, depending on which one is able to handle this particular type of
         * body.
         *
         * @param emptyBodyVisitor          The visitor function for handling objects of the type `EmptyBody`
         * @param conjunctiveBodyVisitor    The visitor function for handling objects of the type `ConjunctiveBody`
         */
        virtual void visit(EmptyBodyVisitor emptyBodyVisitor, ConjunctiveBodyVisitor conjunctiveBodyVisitor) const = 0;

};
