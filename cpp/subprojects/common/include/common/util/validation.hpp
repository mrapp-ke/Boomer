/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include <stdexcept>
#include <string>

/**
 * Throws a `std::invalid_argument` exception if a given value is not greater than a specific threshold.
 *
 * @tparam T        The type of the value and threshold
 * @param name      The name of the parameter, the value corresponds to
 * @param value     The value
 * @param threshold The threshold
 */
template<typename T>
static inline constexpr void assertGreater(const std::string& name, const T value, const T threshold) {
    if (value <= threshold) {
        throw std::invalid_argument("Invalid value given for parameter \"" + name + "\": Must be greater than "
                                    + std::to_string(threshold) + ", but is " + std::to_string(value));
    }
}

/**
 * Throws a `std::invalid_argument` exception if a given value not greater or equal to a specific threshold.
 *
 * @tparam T        The type of the value and threshold
 * @param name      The name of the parameter, the value corresponds to
 * @param value     The value
 * @param threshold The threshold
 */
template<typename T>
static inline constexpr void assertGreaterOrEqual(const std::string& name, const T value, const T threshold) {
    if (value < threshold) {
        throw std::invalid_argument("Invalid value given for parameter \"" + name + "\": Must be greater or equal to "
                                    + std::to_string(threshold) + ", but is " + std::to_string(value));
    }
}

/**
 * Throws a `std::invalid_argument` exception if a given value is not less than a specific threshold.
 *
 * @tparam T        The type of the value and threshold
 * @param name      The name of the parameter, the value corresponds to
 * @param value     The value
 * @param threshold The threshold
 */
template<typename T>
static inline constexpr void assertLess(const std::string& name, const T value, const T threshold) {
    if (value >= threshold) {
        throw std::invalid_argument("Invalid value given for parameter \"" + name + "\": Must be less than "
                                    + std::to_string(threshold) + ", but is " + std::to_string(value));
    }
}

/**
 * Throws a `std::invalid_argument` exception if a given value is not less or equal to a specific threshold.
 *
 * @tparam T        The type of the value and threshold
 * @param name      The name of the parameter, the value corresponds to
 * @param value     The value
 * @param threshold The threshold
 */
template<typename T>
static inline constexpr void assertLessOrEqual(const std::string& name, const T value, const T threshold) {
    if (value > threshold) {
        throw std::invalid_argument("Invalid value given for parameter \"" + name + "\": Must be less or equal to "
                                    + std::to_string(threshold) + ", but is " + std::to_string(value));
    }
}

/**
 * Throws a `std::invalid_argument` exception if a given value is not a multiple of another value.
 *
 * @tparam T    The type of the values
 * @param name  The name of the parameter, the value corresponds to
 * @param value The value that should be a multiple of `other`
 * @param other The other value
 */
template<typename T>
static inline constexpr void assertMultiple(const std::string& name, const T value, const T other) {
    if (value % other != 0) {
        throw std::invalid_argument("Invalid value given for parameter \"" + name + "\": Must be a multiple of "
                                    + std::to_string(other) + ", but is " + std::to_string(value));
    }
}
