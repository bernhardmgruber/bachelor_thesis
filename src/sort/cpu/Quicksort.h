#pragma once

#include <cassert>

#include "../../common/CPUAlgorithm.h"
#include "../SortAlgorithm.h"

namespace cpu
{
    template<typename T>
    class Quicksort : public CPUAlgorithm<T>, public SortAlgorithm
    {
    public:
        const string getName() override
        {
            return "Quicksort";
        }

        bool isInPlace() override
        {
            return true;
        }

        void run(T* data, T* result, size_t size) override
        {
            this->data = data;
            sort_r(0, (long)size - 1);
        }

        virtual ~Quicksort() {}

    private:
        /**
        * Recursive quicksort implementation.
        *
        * @param start
        *            Start index.
        * @param end
        *            End index.
        */
        void sort_r(long start, long end) // do not use size_t as right might become -1 and comparison in while fails
        {
            if (start >= end)
                return;

            long left = start;
            long right = end;

            // select pivot
            T pivot = data[(left + right) / 2];

            // partition array
            while (left <= right)
            {
                while (data[left] < pivot)
                    left++;
                while (pivot < data[right])
                    right--;

                // swap
                if(left <= right)
                {
                    swap(data[right], data[left]);
                    left++;
                    right--;
                }
            }

            // recursion
            sort_r(start, right);
            sort_r(left, end);
        }

        T* data;
    };
}
