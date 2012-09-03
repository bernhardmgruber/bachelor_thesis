#ifndef QSORT_H
#define QSORT_H

#include <cstdlib>

#include "../CPUSortingAlgorithm.h"

using namespace std;

namespace cpu
{
    template<typename T, size_t count>
    class QSort : public CPUSortingAlgorithm<T, count>
    {
        public:
            string getName() override
            {
                return "C stdlib qsort";
            }

            void sort(T* data) override
            {
                qsort(data, count, sizeof(T), [](const void* a, const void* b)
                {
                    return *(T*)a - *(T*)b;
                });
            }

            virtual ~QSort() {}
    };
}

#endif // QSORT_H
