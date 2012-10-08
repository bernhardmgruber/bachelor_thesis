#ifndef PARALLELBITONICSORTB4_H
#define PARALLELBITONICSORTB4_H

#include <algorithm>

#include "../../../common/GPUAlgorithm.h"
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
        class ParallelBitonicSortB4 : public GPUAlgorithm<T>, public SortAlgorithm
        {
            public:
                const string getName()
                {
                    return "Parallel bitonic B4 (Bealto)";
                }

                bool isInPlace() override
                {
                    return false;
                }

                void init(Context* context) override
                {
                    program = context->createProgram("gpu/bealto/ParallelBitonicSortB4.cl", "-D T=" + getTypeName<T>());
                    kernel2 = program->createKernel("ParallelBitonicSortB2");
                    kernel4 = program->createKernel("ParallelBitonicSortB4");
                    delete program;
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
                {
                    buffer = context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * size);
                    queue->enqueueWrite(buffer, data);
                }

                void run(CommandQueue* queue, size_t workGroupSize, size_t size) override
                {
                    for (size_t length=1; length<size; length<<=1)
                    {
                        int inc = length;
                        while (inc > 0)
                        {
                            int ninc = 0;
                            Kernel* kernel;

                            // B4
                            if (inc >= 2 && ninc == 0)
                            {
                                kernel = kernel4;
                                ninc = 2;
                            }
                            else
                                kernel = kernel2;

                            // Always allow B2
                            if (ninc == 0)
                                ninc = 1;
                            size_t nThreads = size >> ninc;
                            workGroupSize = std::min(workGroupSize, nThreads);
                            kernel->setArg(0, buffer);
                            kernel->setArg(1, inc);          // INC passed to kernel
                            kernel->setArg(2, length << 1);  // DIR passed to kernel
                            size_t globalWorkSizes[1] = { nThreads };
                            size_t localWorkSizes[1] = { workGroupSize };
                            queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
                            queue->enqueueBarrier();
                            inc >>= ninc;
                        }
                    }
                }

                void download(CommandQueue* queue, T* result, size_t size) override
                {
                    queue->enqueueRead(buffer, result);
                    delete buffer;
                }

                void cleanup() override
                {
                    delete program;
                    delete kernel2;
                    delete kernel4;
                }

                virtual ~ParallelBitonicSortB4() {}

            private:
                Program* program;
                Kernel* kernel2;
                Kernel* kernel4;
                Buffer* buffer;
        };
    }
}

#endif // PARALLELBITONICSORTB4_H
