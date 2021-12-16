/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"
#include <forward_list>


/**
 * An one-dimensional sparse vector that stores indices in a linked list.
 */
typedef std::forward_list<uint32> BinarySparseListVector;
