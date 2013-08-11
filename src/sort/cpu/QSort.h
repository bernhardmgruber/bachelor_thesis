#pragma once

#include <cstdlib>

#include "../../common/CPUAlgorithm.h"
#include "../SortAlgorithm.h"

using namespace std;

namespace cpu
{
    template<typename T>
    class QSort : public CPUAlgorithm<T>, public SortAlgorithm
    {
    public:
        const string getName() override
        {
            return "C stdlib qsort";
        }

        bool isInPlace() override
        {
            return true;
        }

        void run(T* data, T* result, size_t size) override
        {
            qsort(data, size, sizeof(T), [](const void* a, const void* b) -> int
            {
                if (*((T*)a) < *((T*)b))
                    return -1;
                else if (*((T*)a) > *((T*)b))
                    return 1;
                return 0;
            });
        }

        virtual ~QSort() {}
    };
}
