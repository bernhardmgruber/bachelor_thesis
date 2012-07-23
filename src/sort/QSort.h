#ifndef QSORT_H
#define QSORT_H

#include <cstdlib>

#include "SortingAlgorithm.h"

using namespace std;

template<typename T, size_t size>
class QSort : public SortingAlgorithm<T, size>
{
    public:
        QSort()
            : SortingAlgorithm<T, size>("C stdlib qsort")
        {
        }

        virtual ~QSort()
        {
        }

    protected:
        bool init()
        {
            return true;
        }

        void sort()
        {
            qsort(SortingAlgorithm<T, size>::data, size, sizeof(T), [](const void* a, const void* b) { return *(int*)a - *(int*)b; });
        }

        void cleanup()
        {

        }
};

#endif // QSORT_H
