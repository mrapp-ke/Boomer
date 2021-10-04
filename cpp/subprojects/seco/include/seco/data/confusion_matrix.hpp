/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"


/**
 * A confusion matrix.
 */
struct ConfusionMatrix {

    ConfusionMatrix() : in(0), ip(0), rn(0), rp(0) { };

    /**
     * Irrelevant labels for which a rule predicts negatively.
     */
    float64 in;

    /**
     * Irrelevant labels for which a rule predicts positively.
     */
    float64 ip;

    /**
     * Relevant labels for which a rule predicts negatively.
     */
    float64 rn;

    /**
     * Relevant labels for which a rule predicts positively.
     */
    float64 rp;

    /**
     * Assigns a specific value to the elements of this confusion matrix.
     *
     * @param rhs   A reference to the value to be assigned
     * @return      A reference to the modified confusion matrix
     */
    ConfusionMatrix& operator=(const float64& rhs) {
        in = rhs;
        ip = rhs;
        rn = rhs;
        rp = rhs;
        return *this;
    }

    /**
     * Adds a specific value to the elements of this confusion matrix.
     *
     * @param rhs   A reference to the value to be added
     * @return      A reference to the modified confusion matrix
     */
    ConfusionMatrix& operator+=(const float64& rhs) {
        in += rhs;
        ip += rhs;
        rn += rhs;
        rp += rhs;
        return *this;
    }

    /**
     * Creates and returns a new confusion matrix that results from adding a specific value to the elements of an
     * existing confusion matrix.
     *
     * @param lhs   The original confusion matrix
     * @param rhs   A reference to the value to be added
     * @return      The confusion matrix that has been created
     */
    friend ConfusionMatrix operator+(ConfusionMatrix lhs, const float64& rhs) {
        lhs += rhs;
        return lhs;
    }

    /**
     * Adds the elements of a given confusion matrix to the elements of this confusion matrix.
     *
     * @param rhs   A reference to the confusion matrix, whose elements should be added
     * @return      A reference to the modified confusion matrix
     */
    ConfusionMatrix& operator+=(const ConfusionMatrix& rhs) {
        in += rhs.in;
        ip += rhs.ip;
        rn += rhs.rn;
        rp += rhs.rp;
        return *this;
    }

    /**
     * Creates and returns a new confusion matrix that results from adding the elements of a specific confusion matrix
     * to the elements of an existing confusion matrix.
     *
     * @param lhs   The original confusion matrix
     * @param rhs   A reference to the confusion matrix, whose elements should be added
     * @return      The confusion matrix that has been created
     */
    friend ConfusionMatrix operator+(ConfusionMatrix lhs, const ConfusionMatrix& rhs) {
        lhs += rhs;
        return lhs;
    }

    /**
     * Subtracts a specific value from the elements of this confusion matrix.
     *
     * @param rhs   A reference to the value to be subtracted
     * @return      A reference to the modified confusion matrix
     */
    ConfusionMatrix& operator-=(const float64& rhs) {
        in -= rhs;
        ip -= rhs;
        rn -= rhs;
        rp -= rhs;
        return *this;
    }

    /**
     * Creates and returns a new confusion matrix that results from subtracting a specific value from the elements of an
     * existing confusion matrix.
     *
     * @param lhs   The original confusion matrix
     * @param rhs   A reference to the value to be subtracted
     * @return      The confusion matrix that has been created
     */
    friend ConfusionMatrix operator-(ConfusionMatrix lhs, const float64& rhs) {
        lhs -= rhs;
        return lhs;
    }

    /**
     * Subtracts the elements of a given confusion matrix from the elements of this confusion matrix.
     *
     * @param rhs   A reference to the confusion matrix, whose elements should be subtracted
     * @return      A reference to the modified confusion matrix
     */
    ConfusionMatrix& operator-=(const ConfusionMatrix& rhs) {
        in -= rhs.in;
        ip -= rhs.ip;
        rn -= rhs.rn;
        rp -= rhs.rp;
        return *this;
    }

    /**
     * Creates and returns a new confusion matrix that results from subtracting the elements of a specific confusion
     * matrix from the elements of an existing confusion matrix.
     *
     * @param lhs   The original confusion matrix
     * @param rhs   A reference to the value to be subtracted
     * @return      The confusion matrix that has been created
     */
    friend ConfusionMatrix operator-(ConfusionMatrix lhs, const ConfusionMatrix& rhs) {
        lhs -= rhs;
        return lhs;
    }

    /**
     * Multiplies the elements of this confusion matrix with a specific value.
     *
     * @param rhs   A reference to the value to be multiplied by
     * @return      A reference to the modified confusion matrix
     */
    ConfusionMatrix& operator*=(const float64& rhs) {
        in *= rhs;
        ip *= rhs;
        rn *= rhs;
        rp *= rhs;
        return *this;
    }

    /**
     * Creates and returns a new confusion matrix that results from multiplying the elements of an existing confusion
     * matrix with a specific value.
     *
     * @param lhs   The original confusion matrix
     * @param rhs   A reference to the value to be multiplied by
     * @return      The confusion matrix that has been created
     */
    friend ConfusionMatrix operator*(ConfusionMatrix lhs, const float64& rhs) {
        lhs *= rhs;
        return lhs;
    }

    /**
     * Returns a reference to a confusion matrix element, given the ground truth and a prediction.
     *
     * @param trueLabel     The true label according to the ground truth
     * @param majorityLabel The prediction of the default rule. The prediction is assumed to be the inverse
     * @return              A reference to a confusion matrix element
     */
    float64& getElement(bool trueLabel, bool majorityLabel) {
        if (trueLabel) {
            return majorityLabel ? rn : rp;
        } else {
            return majorityLabel ? in : ip;
        }
    }

};
