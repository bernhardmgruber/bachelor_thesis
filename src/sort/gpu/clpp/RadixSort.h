#ifndef GPUCLPPRADIXSORT_H
#define GPUCLPPRADIXSORT_H

// In order to test that no value has been loosed ! Can take time to check !
#define PARAM_CHECK_HASLOOSEDVALUES 0
#define PARAM_BENCHMARK_LOOPS 20

// The number of bits to sort
#define PARAM_SORT_BITS 32

#include "../../../common/GPUAlgorithm.h"
#include "../../SortAlgorithm.h"

#include "clpp/clppSort_RadixSortGPU.h"

using namespace std;

namespace gpu
{
    namespace clpp
    {
        template<typename T, size_t count>
        class RadixSort : public GPUAlgorithm<T, count>, public SortAlgorithm
        {
            public:
                string getName() override
                {
                    return "Radixsort (clpp)";
                }

                void init(Context* context) override
                {
                    clppProgram::setBasePath("../common/libs/clpp/clpp/");
                    clppcontext.setup(0, 0);

                    s = new clppSort_RadixSortGPU(&clppcontext, count, PARAM_SORT_BITS, true);
                }

                void upload(Context* context, size_t workGroupSize, T* data) override
                {
                    s->pushDatas(data, count);
                }

                void run(CommandQueue* queue, size_t workGroupSize) override
                {
                    s->sort();
                    s->waitCompletion();
                }

                void download(CommandQueue* queue, T* result) override
                {
                    s->popDatas();
                }

                void cleanup() override
                {
                    delete s;
                }

                virtual ~RadixSort() {}

            private:
                clppSort* s;
                clppContext clppcontext;
        };
    }
}

#endif // GPUCLPPRADIXSORT_H
