#ifndef PARALLELBITONICSORTB4_H
#define PARALLELBITONICSORTB4_H

#include <algorithm>

#include "../../GPUSortingAlgorithm.h"
//#include "../../OpenCL.h"

using namespace std;

namespace gpu
{
    namespace bealto
    {
        /**
         * From: http://www.bealto.com/gpu-sorting_intro.html
         */
        template<typename T, size_t count>
        class ParallelBitonicSortB4 : public GPUSortingAlgorithm<T, count>
        {
            public:
                string getName()
                {
                    return "Parallel bitonic B4 (Bealto)";
                }

                void init(Context* context) override
                {
                    program = context->createProgram("gpu/bealto/ParallelBitonicSortB4.cl");
                    kernel2 = program->createKernel("ParallelBitonicSortB2");
                    kernel4 = program->createKernel("ParallelBitonicSortB4");
                }

                void upload(Context* context, T* data) override
                {
                    buffer = context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * count, data);
                }

                void sort(CommandQueue* queue, size_t workGroupSize) override
                {
                    for (size_t length=1; length<count; length<<=1)
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
                            size_t nThreads = count >> ninc;
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
                    queue->finish();
                }

                void download(CommandQueue* queue, T* data) override
                {
                    queue->enqueueRead(buffer, data);
                    queue->finish();
                }

                void cleanup() override
                {
                    delete program;
                    delete buffer;
                    delete kernel2;
                    delete kernel4;
                }

            private:
                Program* program;
                Kernel* kernel2;
                Kernel* kernel4;
                Buffer* buffer;
        };
    }
}

#endif // PARALLELBITONICSORTB4_H
