#ifndef QUICKSORT_H
#define QUICKSORT_H

#include "../../common/CPUAlgorithm.h"
#include "../SortAlgorithm.h"

namespace cpu
{
    template<typename T, size_t count>
    class Quicksort : public CPUAlgorithm<T, count>, public SortAlgorithm
    {
        public:
            string getName() override
            {
                return "Quicksort";
            }

            bool isInPlace() override
            {
                return true;
            }

            void run(T* data, T* result) override
            {
                this->data = data;
                sort_r(0, count - 1);
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
            void sort_r(int start, int end)
            {
                if (start >= end)
                    return;

                int left = start;
                int right = end;

                // select pivot
                int pivot = data[(left + right) / 2];

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
#endif // QUICKSORT_H
