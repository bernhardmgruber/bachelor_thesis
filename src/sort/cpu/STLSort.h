#ifndef STLSORT_H
#define STLSORT_H

#include <algorithm>

#include "../CPUSortingAlgorithm.h"

using namespace std;

template<typename T, size_t count>
class STLSort : public CPUSortingAlgorithm<T, count>
{
    public:
        STLSort()
            : CPUSortingAlgorithm<T, count>("C++ STL algorithm sort")
        {
        }

    protected:
        void sort()
        {
            std::sort(SortingAlgorithm<T, count>::data, SortingAlgorithm<T, count>::data + count);
        }
};

#endif // STLSORT_H
