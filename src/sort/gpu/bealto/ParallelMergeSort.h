#pragma once

#include "../../../common/CLAlgorithm.h"
#include "../../SortAlgorithm.h"

using namespace std;

namespace gpu
{
    namespace bealto
    {
        /**
         * From: http://www.bealto.com/gpu-sorting_intro.html
         */
        template<typename T>
        class ParallelMergeSort : public CLAlgorithm<T>, public SortAlgorithm
        {
            public:
                const string getName() override
                {
                    return "Parallel merge (Bealto)";
                }

                bool isInPlace() override
                {
                    return false;
                }

                void init() override
                {
                    Program* program = context->createProgram("gpu/bealto/ParallelMergeSort.cl", "-D T=" + getTypeName<T>());
                    kernel = program->createKernel("ParallelMergeSort");
                    delete program;
                }

                void upload(size_t workGroupSize, T* data, size_t size) override
                {
                    in = context->createBuffer(CL_MEM_READ_ONLY, sizeof(T) * size);
                    queue->enqueueWrite(in, data);
                    out = context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * size);
                }

                void run(size_t workGroupSize, size_t size) override
                {
                    kernel->setArg(0, in);
                    kernel->setArg(1, out);
                    kernel->setArg(2, sizeof(cl_int) * workGroupSize, nullptr);
                    size_t globalWorkSizes[1] = { size };
                    size_t localWorkSizes[1] = { workGroupSize };
                    queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
                }

                void download(T* result, size_t size) override
                {
                    queue->enqueueRead(out, result);
                    delete in;
                    delete out;
                }

                void cleanup() override
                {
                    delete kernel;
                }

                virtual ~ParallelMergeSort() {}

            private:
                Kernel* kernel;
                Buffer* in;
                Buffer* out;
        };
    }
}
