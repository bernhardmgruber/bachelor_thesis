#pragma once

#include "timsort.hpp"

#include "../../common/CPUAlgorithm.h"
#include "../SortAlgorithm.h"

using namespace std;

namespace cpu
{
    /**
     * Timsort implementation from https://github.com/swenson/sort
     */
    template<typename T>
    class TimSort : public CPUAlgorithm<T>, public SortAlgorithm
    {
        public:
            const string getName() override
            {
                return "Timsort";
            }

            bool isInPlace() override
            {
                return true;
            }

            void run(T* data, T* result, size_t size) override
            {
                timsort(data, data + size, [](const T& a, const T& b) { return a < b; });
            }

            virtual ~TimSort() {}
    };
}
