#ifndef GPUCLPPRADIXSORT_H
#define GPUCLPPRADIXSORT_H

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

                bool isInPlace() override
                {
                    return true;
                }

                void init(Context* context) override
                {
                    clppProgram::setBasePath("../common/libs/clpp/clpp/");
                    clppcontext.setup(0, 0);

                    s = new clppSort_RadixSortGPU(&clppcontext, count, sizeof(T) * 8, true);
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data) override
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
