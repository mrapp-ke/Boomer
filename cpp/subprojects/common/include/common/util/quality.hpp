/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"

/**
 * A struct that stores a numerical score that represents a quality.
 */
struct Quality {
    public:

        Quality() {};

        /**
         * @param q A numerical score that represents the quality
         */
        Quality(float64 q) : quality(q) {};

        /**
         * @param q A reference to an object of type `Quality` to be copied
         */
        Quality(const Quality& q) : quality(q.quality) {};

        /**
         * Assigns the numerical score of an existing object to this object.
         *
         * @param q A reference to the existing object
         * @return  A reference to the modified object
         */
        Quality& operator=(const Quality& q) {
            quality = q.quality;
            return *this;
        }

        /**
         * A numerical score that represents the quality.
         */
        float64 quality;
};
