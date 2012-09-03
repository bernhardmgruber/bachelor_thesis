#ifndef STLSORT_H
#define STLSORT_H

#include <algorithm>

#include "../CPUSortingAlgorithm.h"

using namespace std;

namespace cpu
{
    template<typename T, size_t count>
    class STLSort : public CPUSortingAlgorithm<T, count>
    {
        public:
            void sort(T* data) override
            {
                std::sort(data, data + count);
            }

            string getName() override
            {
                return "C++ STL algorithm sort";
            }

            virtual ~STLSort() {}
    };
}

#endif // STLSORT_H
