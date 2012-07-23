#ifndef QUICKSORT_H
#define QUICKSORT_H

#include "SortingAlgorithm.h"

template<typename T, size_t size>
class Quicksort : public SortingAlgorithm<T, size>
{
    public:
        Quicksort()
            : SortingAlgorithm<T, size>("Quicksort")
        {
        }

        virtual ~Quicksort()
        {
        }

    protected:
        bool init()
        {
            return true;
        }

        void sort()
        {
            sort_r(0, size - 1);
        }

        void cleanup()
        {

        }

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
            int pivot = SortingAlgorithm<T, size>::data[(left + right) / 2];

            // partition array
            while (left <= right)
            {
                while (SortingAlgorithm<T, size>::data[left] < pivot)
                    left++;
                while (pivot < SortingAlgorithm<T, size>::data[right])
                    right--;

                // swap
                if (left <= right)
                {
                    int temp = SortingAlgorithm<T, size>::data[right];
                    SortingAlgorithm<T, size>::data[right] = SortingAlgorithm<T, size>::data[left];
                    SortingAlgorithm<T, size>::data[left] = temp;

                    left++;
                    right--;
                }
            }

            // recursion
            sort_r(start, left - 1);
            sort_r(left, end);
        }
};

#endif // QUICKSORT_H
