#ifndef PARALLELBITONICSORTB16_H
#define PARALLELBITONICSORTB16_H

#include <algorithm>

#include "../../GPUSortingAlgorithm.h"
//#include "../../OpenCL.h"

using namespace std;

/**
 * From: http://www.bealto.com/gpu-sorting_intro.html
 */
template<typename T, size_t count>
class ParallelBitonicSortB16 : public GPUSortingAlgorithm<T, count>
{
    using Base = GPUSortingAlgorithm<T, count>;

public:
    ParallelBitonicSortB16(Context* context, CommandQueue* queue)
        : GPUSortingAlgorithm<T, count>("Parallel bitonic B16 (Bealto)", context, queue, true)
    {
    }

    virtual ~ParallelBitonicSortB16()
    {
    }

protected:
    bool init()
    {
        program = Base::context->createProgram("gpu/bealto/ParallelBitonicSortB16.cl");
        kernel2 = program->createKernel("ParallelBitonicSortB2");
        kernel4 = program->createKernel("ParallelBitonicSortB4");
        kernel8 = program->createKernel("ParallelBitonicSortB8");
        kernel16 = program->createKernel("ParallelBitonicSortB16");

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

                if (inc >= 8 && ninc == 0)
                {
                  kernel = kernel16;
                  ninc = 4;
                }

                if (inc >= 4 && ninc == 0)
                {
                  kernel = kernel8;
                  ninc = 3;
                }

                if (inc >= 2 && ninc == 0)
                {
                  kernel = kernel4;
                  ninc = 2;
                }

                if (ninc == 0)
                {
                  kernel = kernel2;
                  ninc = 1;
                }

                size_t kernelWorkGroupSize = kernel->getWorkGroupSize();
                if(workGroupSize > kernelWorkGroupSize)
                {
                    cerr << "Skipped kernel for WG size " << workGroupSize <<  " (max is " << kernelWorkGroupSize << ") . Out of resources." << endl;
                    return;
                }

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
        delete kernel8;
        delete kernel16;
    }

    Program* program;
    Kernel* kernel2;
    Kernel* kernel4;
    Kernel* kernel8;
    Kernel* kernel16;
    Buffer* buffer;
};

#endif // PARALLELBITONICSORTB16_H
