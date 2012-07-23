#ifndef STLSORT_H
#define STLSORT_H

#include <algorithm>

#include "SortingAlgorithm.h"

using namespace std;

template<typename T, size_t count>
class STLSort : public SortingAlgorithm<T, count>
{
    public:
        STLSort()
            : SortingAlgorithm<T, count>("C++ STL algorithm sort")
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
            std::sort(SortingAlgorithm<T, count>::data, SortingAlgorithm<T, count>::data + count);
        }

        void cleanup()
        {

        }
};

#endif // STLSORT_H
