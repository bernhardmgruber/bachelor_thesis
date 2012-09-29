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
        template<typename T>
        class RadixSort : public GPUAlgorithm<T>, public SortAlgorithm
        {
            public:
                const string getName() override
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

                    s = new clppSort_RadixSortGPU(&clppcontext, 0, sizeof(T) * 8, true);
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
                {
                    s->pushDatas(data, size);
                }

                void run(CommandQueue* queue, size_t workGroupSize, size_t size) override
                {
                    s->sort();
                    s->waitCompletion();
                }

                void download(CommandQueue* queue, T* result, size_t size) override
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
