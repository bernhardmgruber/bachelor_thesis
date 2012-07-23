#ifndef STLSORT_H
#define STLSORT_H

#include <algorithm>

#include "SortingAlgorithm.h"

using namespace std;

template<typename T, size_t size>
class STLSort : public SortingAlgorithm<T, size>
{
    public:
        STLSort()
            : SortingAlgorithm<T, size>("C++ STL algorithm sort")
        {
        }

        virtual ~STLSort()
        {
        }

    protected:
        bool init()
        {
            return true;
        }

        void sort()
        {
            std::sort(SortingAlgorithm<T, size>::data, SortingAlgorithm<T, size>::data + size);
        }

        void cleanup()
        {

        }
};

#endif // STLSORT_H
