/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/matrix_sparse_set.hpp"
#include "common/data/view_c_contiguous.hpp"
#include "common/statistics/statistics.hpp"

#include <functional>

/**
 * Defines an interface for all classes that provide access to gradients and Hessians which serve as the basis for
 * learning a new boosted rule or refining an existing one.
 */
class IBoostingStatistics : public IStatistics {
    public:

        virtual ~IBoostingStatistics() {};

        /**
         * A visitor function for handling score matrices of the type `CContiguousConstView`.
         */
        typedef std::function<void(const CContiguousConstView<float64>&)> DenseScoreMatrixVisitor;

        /**
         * A visitor function for handling score matrices of the type `SparseSetMatrix`.
         */
        typedef std::function<void(const SparseSetMatrix<float64>&)> SparseScoreMatrixVisitor;

        /**
         * Invokes one of the given visitor functions, depending on which one is able to handle the type of matrix that
         * is used to store the currently predicted scores.
         *
         * @param denseVisitor  The visitor function for handling objects of the type `CContiguousConstView`
         * @param sparseVisitor The visitor function for handling objects of the type `SparseSetMatrix`
         */
        virtual void visitScoreMatrix(DenseScoreMatrixVisitor denseVisitor,
                                      SparseScoreMatrixVisitor sparseVisitor) const = 0;
};
