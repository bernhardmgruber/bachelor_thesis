#ifndef PARALLELBITONICSORTC_H
#define PARALLELBITONICSORTC_H

#include <list>

#include "../../../common/GPUAlgorithm.h"
#include "../../SortAlgorithm.h"

#define ALLOWB (16 + 8 + 4 + 2)

using namespace std;

namespace gpu
{
    namespace bealto
    {
        /**
         * From: http://www.bealto.com/gpu-sorting_intro.html
         */
        template<typename T>
        class ParallelBitonicSortC : public GPUAlgorithm<T>, public SortAlgorithm
        {

            public:
                const string getName() override
                {
                    return "Parallel bitonic C (Bealto)";
                }

                bool isInPlace() override
                {
                    return false;
                }

                void init(Context* context) override
                {
                    program = context->createProgram("gpu/bealto/ParallelBitonicSortC.cl", "-I gpu/bealto/ -D T=" + getTypeName<T>());
                    kernel2 = program->createKernel("ParallelBitonicSortB2");
                    kernel4 = program->createKernel("ParallelBitonicSortB4");
                    kernel8 = program->createKernel("ParallelBitonicSortB8");
                    kernel16 = program->createKernel("ParallelBitonicSortB16");
                    kernelC4 = program->createKernel("ParallelBitonicSortC4");
                    delete program;
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
                {
                    buffer = context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * size);
                    queue->enqueueWrite(buffer, data);
                }

                void run(CommandQueue* queue, size_t workGroupSize, size_t size) override
                {
                    //int mLastN = -1;

                    for (size_t length=1; length<size; length<<=1)
                    {
                        int inc = length;
                        std::list<int> strategy; // vector defining the sequence of reductions
                        {
                            int ii = inc;
                            while (ii>0)
                            {
                                if (ii==128 || ii==32 || ii==8)
                                {
                                    strategy.push_back(-1);    // C kernel
                                    break;
                                }
                                int d = 1; // default is 1 bit
                                if (0) d = 1;
#if 1
                                // Force jump to 128
                                else if (ii==256) d = 1;
                                else if (ii==512 && (ALLOWB & 4)) d = 2;
                                else if (ii==1024 && (ALLOWB & 8)) d = 3;
                                else if (ii==2048 && (ALLOWB & 16)) d = 4;
#endif
                                else if (ii>=8 && (ALLOWB & 16)) d = 4;
                                else if (ii>=4 && (ALLOWB & 8)) d = 3;
                                else if (ii>=2 && (ALLOWB & 4)) d = 2;
                                else d = 1;
                                strategy.push_back(d);
                                ii >>= d;
                            }
                        }

                        while (inc > 0)
                        {
                            int ninc = 0;
                            int doLocal = 0;
                            size_t nThreads = 0;
                            int d = strategy.front();
                            strategy.pop_front();

                            Kernel* kernel = nullptr;

                            switch (d)
                            {
                                case -1:
                                    kernel = kernelC4;
                                    ninc = -1; // reduce all bits
                                    doLocal = 4;
                                    nThreads = size >> 2;
                                    break;
                                case 4:
                                    kernel = kernel16;
                                    ninc = 4;
                                    nThreads = size >> ninc;
                                    break;
                                case 3:
                                    kernel = kernel8;
                                    ninc = 3;
                                    nThreads = size >> ninc;
                                    break;
                                case 2:
                                    kernel = kernel4;
                                    ninc = 2;
                                    nThreads = size >> ninc;
                                    break;
                                case 1:
                                    kernel = kernel2;
                                    ninc = 1;
                                    nThreads = size >> ninc;
                                    break;
                                default:
                                    printf("Strategy error!\n");
                                    break;
                            }
                            size_t wg = workGroupSize;
                            wg = std::min(wg, nThreads);
                            kernel->setArg(0, buffer);
                            kernel->setArg(1, inc); // INC passed to kernel
                            kernel->setArg(2, (int)(length << 1)); // DIR passed to kernel

                            if (doLocal > 0) // DOLOCAL values / thread
                                kernel->setArg(3, doLocal * wg * sizeof(cl_int), nullptr);

                            size_t globalWorkSizes[1] = { nThreads };
                            size_t localWorkSizes[1] = { wg };
                            queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
                            queue->enqueueBarrier();
                            if (ninc < 0)
                                break; // done
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
                    delete kernel8;
                    delete kernel16;
                    delete kernelC4;
                }

                virtual ~ParallelBitonicSortC() {}

            private:
                Program* program;
                Kernel* kernel2;
                Kernel* kernel4;
                Kernel* kernel8;
                Kernel* kernel16;
                Kernel* kernelC4;
                Buffer* buffer;
        };
    }
}

#endif // PARALLELBITONICSORTC_H
