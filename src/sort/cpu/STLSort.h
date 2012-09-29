#ifndef STLSORT_H
#define STLSORT_H

#include <algorithm>

#include "../../common/CPUAlgorithm.h"
#include "../SortAlgorithm.h"

using namespace std;

namespace cpu
{
    template<typename T>
    class STLSort : public CPUAlgorithm<T>, public SortAlgorithm
    {
        public:
            const string getName() override
            {
                return "C++ STL algorithm sort";
            }

            bool isInPlace() override
            {
                return true;
            }

            void run(T* data, T* result, size_t size) override
            {
                std::sort(data, data + size);
            }

            virtual ~STLSort() {}
    };
}

#endif // STLSORT_H
