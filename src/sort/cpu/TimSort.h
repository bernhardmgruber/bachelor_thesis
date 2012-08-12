#ifndef TIMSORT_H
#define TIMSORT_H

#include "timsort.hpp"

#include "../CPUSortingAlgorithm.h"

using namespace std;

/**
 * Timsort implementation from https://github.com/swenson/sort
 */
template<typename T, size_t count>
class TimSort : public CPUSortingAlgorithm<T, count>
{
    public:
        TimSort()
            : CPUSortingAlgorithm<T, count>("Timsort")
        {
        }

        virtual ~TimSort()
        {
        }

    protected:
        bool init()
        {
            return true;
        }

        void sort()
        {
            timsort(SortingAlgorithm<T, count>::data, SortingAlgorithm<T, count>::data + count, [](const T& a, const T& b) { return a < b; });
        }

        void cleanup()
        {
        }
};

#endif // TIMSORT_H
