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
                using Base = GPUSortingAlgorithm<T, count>;

            public:
                ParallelBitonicSortB4(Context* context, CommandQueue* queue)
                    : GPUSortingAlgorithm<T, count>("Parallel bitonic B4 (Bealto)", context, queue, true)
                {
                }

                virtual ~ParallelBitonicSortB4()
                {
                }

            protected:
                bool init()
                {
                    program = Base::context->createProgram("gpu/bealto/ParallelBitonicSortB4.cl");
                    kernel2 = program->createKernel("ParallelBitonicSortB2");
                    kernel4 = program->createKernel("ParallelBitonicSortB4");

                    return true;
                }

                void upload()
                {
                    buffer = Base::context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * count);
                    Base::queue->enqueueWrite(buffer, SortingAlgorithm<T, count>::data);
                    Base::queue->finish();
                }

                void sort(size_t workGroupSize)
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
                            Base::queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
                            Base::queue->enqueueBarrier();
                            inc >>= ninc;
                        }
                    }
                    Base::queue->finish();
                }

                void download()
                {
                    Base::queue->enqueueRead(buffer, SortingAlgorithm<T, count>::data);
                    Base::queue->finish();
                }

                void cleanup()
                {
                    delete program;
                    delete buffer;
                    delete kernel2;
                    delete kernel4;
                }

                Program* program;
                Kernel* kernel2;
                Kernel* kernel4;
                Buffer* buffer;
        };
    }
}

#endif // PARALLELBITONICSORTB4_H
