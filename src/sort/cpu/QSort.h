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
                    return (int)(*(T*)a - *(T*)b);
                });
            }

            virtual ~QSort() {}
    };

    template<>
    void QSort<double>::run(double* data, double* result, size_t size)
    {
        qsort(data, size, sizeof(double), [](const void* a, const void* b)
        {
            if (*((double*)a) < *((double*)b))
                return -1;
            else if (*((double*)a) > *((double*)b))
                return 1;
            return 0;
        });
    }

    template<>
    void QSort<float>::run(float* data, float* result, size_t size)
    {
        qsort(data, size, sizeof(float), [](const void* a, const void* b)
        {
            if (*((float*)a) < *((float*)b))
                return -1;
            else if (*((float*)a) > *((float*)b))
                return 1;
            return 0;
        });
    }
}

#endif // QSORT_H
