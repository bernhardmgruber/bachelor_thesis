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
            data = new T[size];
            for(size_t i = 0; i < size; i++)
                data[i] = rand();

            return true;
        }

        void sort()
        {
            sort_r(0, size - 1);
        }

        void cleanup()
        {
            delete[] data;
        }

    private:
        int* data;

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
                    int temp = data[right];
                    data[right] = data[left];
                    data[left] = temp;

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
