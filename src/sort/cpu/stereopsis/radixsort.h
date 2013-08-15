#pragma once

#include <cstdlib>

#include "../../../common/CPUAlgorithm.h"
#include "../../SortAlgorithm.h"

using namespace std;

namespace cpu
{
    namespace stereopsis
    {
        template<typename T>
        class RadixSort : public CPUAlgorithm<T>, public SortAlgorithm
        {
        public:
            static const unsigned int RADIX_LENGTH = 16;
            static const unsigned int HISTOGRAM_BUCKETS = (1 << RADIX_LENGTH);
            static const unsigned int RADIX_MASK = (HISTOGRAM_BUCKETS - 1);

            static const unsigned int RUNS = (sizeof(T) * 8)  % RADIX_LENGTH == 0 ? (sizeof(T) * 8) / RADIX_LENGTH : (sizeof(T) * 8) / RADIX_LENGTH + 1;

            const string getName() override
            {
                return "Radixsort (stereopsis)";
            }

            bool isInPlace() override
            {
                return !(RUNS & 1u); // in place if RUNS is even
            }

            void run(T* data, T* result, size_t size) override
            {
                size_t* histograms = new size_t[RUNS * HISTOGRAM_BUCKETS];
                memset(histograms, 0, RUNS * HISTOGRAM_BUCKETS * sizeof(size_t));

                // 1.  parallel histogramming pass
                for (size_t i = 0; i < size; i++) 
                {
                    T element = data[i];
                    
                    for(int r = 0; r < RUNS; r++)
                    {
                        T pos = (element >> (r * RADIX_LENGTH)) & RADIX_MASK;
                        histograms[r * HISTOGRAM_BUCKETS + pos]++;
                    }
                }

                // 2.  Sum the histograms -- each histogram entry records the number of values preceding itself.
                for(unsigned int r = 0; r < RUNS; r++) 
                {
                    size_t sum = 0;
                    for (unsigned int i = 0; i < HISTOGRAM_BUCKETS; i++) 
                    {
                        size_t val = histograms[r * HISTOGRAM_BUCKETS + i];
                        histograms[r * HISTOGRAM_BUCKETS + i] = sum;
                        sum += val;
                    }
                }

                T* src = data;
                T* dst = result;
                for(unsigned int r = 0; r < RUNS; r++) 
                {
                    for (size_t i = 0; i < size; i++) 
                    {
                        T element = src[i];
                        T pos = ((element >> (r * RADIX_LENGTH)) & RADIX_MASK);

                        size_t& index = histograms[r * HISTOGRAM_BUCKETS + pos];
                        dst[index] = element;
                        index++;
                    }

                    swap(src, dst);
                }

                delete[] histograms;
            }

            virtual ~RadixSort() {}
        };
    }
}
