#ifndef QSORT_H
#define QSORT_H

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
                qsort(data, size, sizeof(T), [](const void* a, const void* b)
                {
                    return *(T*)a - *(T*)b;
                });
            }

            virtual ~QSort() {}
    };
}

#endif // QSORT_H
