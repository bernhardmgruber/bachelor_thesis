#pragma once

#include <algorithm>

#include "../../../common/CLAlgorithm.h"
#include "../../SortAlgorithm.h"

using namespace std;

namespace gpu
{
    namespace dixxi
    {
        /**
         * From: http://www.bealto.com/gpu-sorting_intro.html
         * ParallelBitonicSortB2
         */
        template<typename T>
        class BitonicSort : public CLAlgorithm<T>, public SortAlgorithm
        {
            public:
                const string getName() override
                {
                    return "Bitonic sort (dixxi Bealto)";
                }

                bool isInPlace() override
                {
                    return false;
                }

                void init() override
                {
                    Program* program = context->createProgram("gpu/dixxi/BitonicSort.cl", "-D T=" + getTypeName<T>());
                    kernel = program->createKernel("BitonicSort");
                    delete program;
                }

                void upload(size_t workGroupSize, T* data, size_t size) override
                {
                    buffer = context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * size);
                    queue->enqueueWrite(buffer, data);
                }

                void run(size_t workGroupSize, size_t size) override
                {
                    for (cl_int box = 1; box < size; box <<= 1)
                    {
                        for (cl_int distance = box; distance > 0; distance >>= 1)
                        {
                            kernel->setArg(0, buffer);
                            kernel->setArg(1, distance);          
                            kernel->setArg(2, box << 1);  

                            size_t globalWorkSizes[1] = { size / 2 };
                            size_t localWorkSizes[1] = { min(workGroupSize, size / 2) };

                            queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
                        }
                    }
                }

                void download(T* result, size_t size) override
                {
                    queue->enqueueRead(buffer, result);
                    delete buffer;
                }

                void cleanup() override
                {
                    delete kernel;
                }

                virtual ~BitonicSort() {}

            private:
                Kernel* kernel;
                Buffer* buffer;
        };
    }
}
