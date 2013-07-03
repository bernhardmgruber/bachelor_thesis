#pragma once

#include "../../../common/CLAlgorithm.h"
#include "../../SortAlgorithm.h"

#include "clpp/clppSort_RadixSortGPU.h"

using namespace std;

namespace gpu
{
    namespace clpp
    {
        template<typename T>
        class RadixSort : public CLAlgorithm<T>, public SortAlgorithm
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

                void init() override
                {
                    clppProgram::setBasePath("../common/libs/clpp/clpp/");
                    clppcontext.setup(0, 0);

                    s = new clppSort_RadixSortGPU(&clppcontext, 0, sizeof(T) * 8, true);
                }

                void upload(size_t workGroupSize, T* data, size_t size) override
                {
                    s->pushDatas(data, size);
                }

                void run(size_t workGroupSize, size_t size) override
                {
                    s->sort();
                }

                void download(T* result, size_t size) override
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

