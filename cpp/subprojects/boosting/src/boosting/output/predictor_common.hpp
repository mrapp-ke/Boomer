#pragma once
#include "common/model/head_complete.hpp"
#include "common/model/head_partial.hpp"


namespace boosting {

    static inline void applyCompleteHead(const CompleteHead& head, CContiguousView<float64>::value_iterator iterator) {
        CompleteHead::score_const_iterator scoreIterator = head.scores_cbegin();
        uint32 numElements = head.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            iterator[i] += scoreIterator[i];
        }
    }

    static inline void applyPartialHead(const PartialHead& head, CContiguousView<float64>::value_iterator iterator) {
        PartialHead::score_const_iterator scoreIterator = head.scores_cbegin();
        PartialHead::index_const_iterator indexIterator = head.indices_cbegin();
        uint32 numElements = head.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];
            iterator[index] += scoreIterator[i];
        }
    }

    static inline void applyHead(const IHead& head, CContiguousView<float64>::value_iterator scoreIterator) {
        auto completeHeadVisitor = [=](const CompleteHead& head) {
            applyCompleteHead(head, scoreIterator);
        };
        auto partialHeadVisitor = [=](const PartialHead& head) {
            applyPartialHead(head, scoreIterator);
        };
        head.visit(completeHeadVisitor, partialHeadVisitor);
    }

    static inline void applyRule(const RuleList::Rule& rule,
                                 CContiguousConstView<const float32>::value_const_iterator featureValuesBegin,
                                 CContiguousConstView<const float32>::value_const_iterator featureValuesEnd,
                                 CContiguousView<float64>::value_iterator scoreIterator) {
        const IBody& body = rule.getBody();

        if (body.covers(featureValuesBegin, featureValuesEnd)) {
            const IHead& head = rule.getHead();
            applyHead(head, scoreIterator);
        }
    }

    static inline void applyRules(const RuleList& model,
                                  CContiguousConstView<const float32>::value_const_iterator featureValuesBegin,
                                  CContiguousConstView<const float32>::value_const_iterator featureValuesEnd,
                                  CContiguousView<float64>::value_iterator scoreIterator) {
        for (auto it = model.used_cbegin(); it != model.used_cend(); it++) {
            const RuleList::Rule& rule = *it;
            applyRule(rule, featureValuesBegin, featureValuesEnd, scoreIterator);
        }
    }

    static inline void applyRuleCsr(const RuleList::Rule& rule,
                                    CsrConstView<const float32>::index_const_iterator featureIndicesBegin,
                                    CsrConstView<const float32>::index_const_iterator featureIndicesEnd,
                                    CsrConstView<const float32>::value_const_iterator featureValuesBegin,
                                    CsrConstView<const float32>::value_const_iterator featureValuesEnd,
                                    CContiguousView<float64>::value_iterator scoreIterator, float32* tmpArray1,
                                    uint32* tmpArray2, uint32 n) {
        const IBody& body = rule.getBody();

        if (body.covers(featureIndicesBegin, featureIndicesEnd, featureValuesBegin, featureValuesEnd, tmpArray1,
                        tmpArray2, n)) {
            const IHead& head = rule.getHead();
            applyHead(head, scoreIterator);
        }
    }

    static inline void applyRulesCsr(const RuleList& model, uint32 numFeatures,
                                     CsrConstView<const float32>::index_const_iterator featureIndicesBegin,
                                     CsrConstView<const float32>::index_const_iterator featureIndicesEnd,
                                     CsrConstView<const float32>::value_const_iterator featureValuesBegin,
                                     CsrConstView<const float32>::value_const_iterator featureValuesEnd,
                                     CContiguousView<float64>::value_iterator scoreIterator) {
        float32* tmpArray1 = new float32[numFeatures];
        uint32* tmpArray2 = new uint32[numFeatures] {};
        uint32 n = 1;

        for (auto it = model.used_cbegin(); it != model.used_cend(); it++) {
            const RuleList::Rule& rule = *it;
            applyRuleCsr(rule, featureIndicesBegin, featureIndicesEnd, featureValuesBegin, featureValuesEnd,
                         scoreIterator, &tmpArray1[0], &tmpArray2[0], n);
            n++;
        }

        delete[] tmpArray1;
        delete[] tmpArray2;
    }

}
