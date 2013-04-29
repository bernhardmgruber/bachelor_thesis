#ifndef STEREOPSISRADIXSORT_H
#define STEREOPSISRADIXSORT_H

#include <cstdlib>

#include "../../common/CPUAlgorithm.h"
#include "../SortAlgorithm.h"

using namespace std;

#define PREFETCH 1

#if PREFETCH
#include <xmmintrin.h>	// for prefetch
#define pfval	64
#define pfval2	128
#define prefetch(x)	_mm_prefetch(cpointer(x + i + pfval), 0)
#define prefetch2(x)	_mm_prefetch(cpointer(x + i + pfval2), 0)
#else
#define prefetch(x)
#define prefetch2(x)
#endif

namespace cpu
{
    namespace stereopsis
    {
        template<typename T>
        class RadixSort : public CPUAlgorithm<T>, public SortAlgorithm
        {
            public:
                static const unsigned int RADIX_LENGTH = 11;
                static const unsigned int HISTOGRAM_BUCKETS = (1 << RADIX_LENGTH);
                static const unsigned int RADIX_MASK = (RADIX_LENGTH - 1);

                static const unsigned int RUNS = roundToMultiple(sizeof(T) * 8, RADIX_LENGTH);

                const string getName() override
                {
                    return "Radixsort (stereopsis)";
                }

                bool isInPlace() override
                {
                    return !(RUNS & 1u); // in place of RUNS is even
                }

                void run(T* data, T* result, size_t size) override
                {
                    unsigned int* histograms = new unsigned int[RUNS * HISTOGRAM_BUCKETS];

                    memset(histograms, 0, RUNS * HISTOGRAM_BUCKETS * sizeof(unsigned int));

	                  // 1.  parallel histogramming pass
	                  for (i = 0; i < size; i++) {
                      prefetch(data);

		                  //uint32 fi = FloatFlip((uint32&)array[i]);

                      for(int r = 0; r < RUNS; r++)
                        histograms[r * HISTOGRAM_BUCKETS + ((data[i] >> (RUNS * RADIX_LENGTH)) & RADIX_MASK)]++;
	                  }
	
	                  // 2.  Sum the histograms -- each histogram entry records the number of values preceding itself.
	
                    unsigned int* sums = new unsigned int[RUNS];
                    memset(sums, 0, RUNS * sizeof(unsigned int));

		                for (i = 0; i < HISTOGRAM_BUCKETS; i++) {
                      for(int r = 0; r < RUNS; r++) {
                        unsigned int tmp = historgrams[r * HISTOGRAM_BUCKETS + i] + sum[r];
                        historgrams[r * HISTOGRAM_BUCKETS + i] = sum[r] - 1;
                        sum[r] = tmp;
                      }
		                }

	                  // floatflip entire value on first run, read/write histogram
                    T* src = data;
                    T* dst = result;
                    for(size_t r = 0; r < RUNS; r++) {
	                    for (i = 0; i < size; i++) {

                        T element = src[i];
		
                        //if(r == 0)
		                    //  FloatFlipX(fi);

		                    size_t pos = ((element >> (RUNS * RADIX_LENGTH)) & RADIX_MASK);
		
		                    prefetch2(src);
		                    dst[++historgrams[r * HISTOGRAM_BUCKETS + pos]] = element;
	                    }

                      swap(src, dst);
                    }

                    delete[] histograms;
                    delete[] sums;
                }

                virtual ~QSort() {}
        };
    }
}

#endif // STEREOPSISRADIXSORT_H
