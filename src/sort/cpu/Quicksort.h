#pragma once

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
                sort_r(0, size - 1);
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
            void sort_r(size_t start, size_t end)
            {
                if (start >= end)
                    return;

                size_t left = start;
                size_t right = end;

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
                    if (left <= right)
                    {
                        swap(data[right], data[left]);

                        left++;
                        right--;
                    }
                }

                // recursion
                sort_r(start, left - 1);
                sort_r(left, end);
            }

            T* data;
    };
}
