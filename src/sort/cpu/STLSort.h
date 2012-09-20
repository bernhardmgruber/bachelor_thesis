#ifndef STLSORT_H
#define STLSORT_H

#include <algorithm>

#include "../../common/CPUAlgorithm.h"
#include "../SortAlgorithm.h"

using namespace std;

namespace cpu
{
    template<typename T, size_t count>
    class STLSort : public CPUAlgorithm<T, count>, public SortAlgorithm
    {
        public:
            string getName() override
            {
                return "C++ STL algorithm sort";
            }

            bool isInPlace() override
            {
                return true;
            }

            void run(T* data, T* result) override
            {
                std::sort(data, data + count);
            }

            virtual ~STLSort() {}
    };
}

#endif // STLSORT_H
