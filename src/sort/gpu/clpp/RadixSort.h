#ifndef GPUCLPPRADIXSORT_H
#define GPUCLPPRADIXSORT_H

// In order to test that no value has been loosed ! Can take time to check !
#define PARAM_CHECK_HASLOOSEDVALUES 0
#define PARAM_BENCHMARK_LOOPS 20

// The number of bits to sort
#define PARAM_SORT_BITS 32

#include "../../GPUSortingAlgorithm.h"

#include "../../libs/clpp/clpp/clppSort_RadixSortGPU.h"

using namespace std;

namespace gpu
{
    namespace clpp
    {
        template<typename T, size_t count>
        class RadixSort : public GPUSortingAlgorithm<T, count>
        {
                using Base = GPUSortingAlgorithm<T, count>;

            public:
                RadixSort(Context* context, CommandQueue* queue)
                    : GPUSortingAlgorithm<T, count>("Radixsort (clpp)", context, queue)
                {
                    clppProgram::setBasePath("libs/clpp/clpp/");

                    clppcontext.setup(0, 0);
                }

                virtual ~RadixSort()
                {
                }

            protected:
                clppSort* s;
                clppContext clppcontext;

                bool init()
                {
                    s = new clppSort_RadixSortGPU(&clppcontext, count, PARAM_SORT_BITS, true);

                    assert(s->_context->clQueue != 0);

                    return true;
                }

                void upload()
                {
                    assert(s->_context->clQueue != 0);
                    s->pushDatas(SortingAlgorithm<T, count>::data, count);
                }

                void sort(size_t workGroupSize)
                {
                    s->sort();
                    s->waitCompletion();
                }

                void download()
                {
                    s->popDatas();
                }

                void cleanup()
                {
                    delete s;
                }
        };
    }
}

#endif // GPUCLPPRADIXSORT_H
