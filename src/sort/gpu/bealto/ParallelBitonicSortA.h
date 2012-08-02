#ifndef PARALLELBITONICSORTA_H
#define PARALLELBITONICSORTA_H

#include "../../GPUSortingAlgorithm.h"
#include "../../OpenCL.h"

using namespace std;

/**
 * From: http://www.bealto.com/gpu-sorting_intro.html
 */
template<typename T, size_t count>
class ParallelBitonicSortA : public GPUSortingAlgorithm<T, count>
{
    using Base = GPUSortingAlgorithm<T, count>;

public:
    ParallelBitonicSortA(Context* context, CommandQueue* queue)
        : GPUSortingAlgorithm<T, count>("Parallel bitonic A (Bealto)", context, queue)
    {
    }

    virtual ~ParallelBitonicSortA()
    {
    }

protected:
    bool init()
    {
        program = Base::context->createProgram("gpu/bealto/ParallelBitonicSortA.cl");
        kernel = program->createKernel("ParallelBitonicSortA");

        return true;
    }

    void upload()
    {
        in = Base::context->createBuffer(CL_MEM_READ_ONLY, sizeof(T) * count);
        out = Base::context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * count);

        Base::queue->enqueueWrite(in, SortingAlgorithm<T, count>::data);
        Base::queue->finish();
    }

    void sort(size_t workGroupSize)
    {
        //int nk = 0;

        bool swapBuffers = false;

        for (size_t length = 1; length < count; length <<= 1)
            for (size_t inc = length; inc > 0; inc >>= 1)
            {
                kernel->setArg(0, swapBuffers ? out : in);
                kernel->setArg(1, swapBuffers ? in : out);
                kernel->setArg(2, inc);
                kernel->setArg(3, length<<1);
                size_t globalWorkSizes[1] = { count };
                size_t localWorkSizes[1] = { workGroupSize };
                Base::queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
                Base::queue->enqueueBarrier();
                swapBuffers = !swapBuffers;
                //nk++;
            }
        Base::queue->finish();
    }

    void download()
    {
        Base::queue->enqueueRead(out, SortingAlgorithm<T, count>::data);
        Base::queue->finish();
    }

    void cleanup()
    {
        delete program;
        delete in;
        delete out;
        delete kernel;
    }

    Program* program;
    Kernel* kernel;
    Buffer* in;
    Buffer* out;
};

#endif // PARALLELBITONICSORTA_H
