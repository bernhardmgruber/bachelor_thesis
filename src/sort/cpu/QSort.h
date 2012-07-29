#ifndef QSORT_H
#define QSORT_H

#include <cstdlib>

#include "../CPUSortingAlgorithm.h"

using namespace std;

template<typename T, size_t count>
class QSort : public CPUSortingAlgorithm<T, count>
{
    using Base = SortingAlgorithm<T, count>;

    public:
        QSort()
            : CPUSortingAlgorithm<T, count>("C stdlib qsort")
        {
        }

    protected:
        void sort()
        {
            qsort(Base::data, count, sizeof(T), [](const void* a, const void* b) { return *(T*)a - *(T*)b; });
        }
};

#endif // QSORT_H
