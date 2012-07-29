#ifndef QSORT_H
#define QSORT_H

#include <cstdlib>

#include "SortingAlgorithm.h"

using namespace std;

template<typename T, size_t count>
class QSort : public SortingAlgorithm<T, count>
{
    public:
        QSort()
            : SortingAlgorithm<T, count>("C stdlib qsort")
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
            qsort(SortingAlgorithm<T, count>::data, count, sizeof(T), [](const void* a, const void* b) { return *(T*)a - *(T*)b; });
        }

        void cleanup()
        {

        }
};

#endif // QSORT_H
