#ifndef TIMSORT_H
#define TIMSORT_H

#include "timsort.hpp"

#include "../CPUSortingAlgorithm.h"

using namespace std;

namespace cpu
{
    /**
     * Timsort implementation from https://github.com/swenson/sort
     */
    template<typename T, size_t count>
    class TimSort : public CPUSortingAlgorithm<T, count>
    {
        public:
            string getName() override
            {
                return "Timsort";
            }

            void sort(T* data) override
            {
                timsort(data, data + count, [](const T& a, const T& b) { return a < b; });
            }

            virtual ~TimSort() {}
    };
}


#endif // TIMSORT_H
