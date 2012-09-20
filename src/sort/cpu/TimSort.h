#ifndef TIMSORT_H
#define TIMSORT_H

#include "timsort.hpp"

#include "../../common/CPUAlgorithm.h"
#include "../SortAlgorithm.h"

using namespace std;

namespace cpu
{
    /**
     * Timsort implementation from https://github.com/swenson/sort
     */
    template<typename T, size_t count>
    class TimSort : public CPUAlgorithm<T, count>, public SortAlgorithm
    {
        public:
            string getName() override
            {
                return "Timsort";
            }

            bool isInPlace() override
            {
                return true;
            }

            void run(T* data, T* result) override
            {
                timsort(data, data + count, [](const T& a, const T& b) { return a < b; });
            }

            virtual ~TimSort() {}
    };
}


#endif // TIMSORT_H
