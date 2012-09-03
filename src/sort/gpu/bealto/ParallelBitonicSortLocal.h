#ifndef PARALLELBITONICSORTLOCAL_H
#define PARALLELBITONICSORTLOCAL_H

#include "../../GPUSortingAlgorithm.h"
//#include "../../OpenCL.h"

using namespace std;

/**
 * From: http://www.bealto.com/gpu-sorting_intro.html
 */
template<typename T, size_t count>
class ParallelBitonicSortLocal : public GPUSortingAlgorithm<T, count>
{
    using Base = GPUSortingAlgorithm<T, count>;

    public:
        string getName() override
        {
            return "Parallel bitonic local (Bealto)";
        }

        void init(Context* context) override
        {
            program = context->createProgram("gpu/bealto/ParallelBitonicSortLocal.cl");
            kernel = program->createKernel("ParallelBitonicSortLocal");
        }

        void upload(Context* context, T* data)
        {
            in = context->createBuffer(CL_MEM_READ_ONLY, sizeof(T) * count, data);
            out = context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * count);
        }

        void sort(CommandQueue* queue, size_t workGroupSize)
        {
            kernel->setArg(0, in);
            kernel->setArg(1, out);
            kernel->setArg(2, sizeof(cl_int) * workGroupSize, nullptr);
            size_t globalWorkSizes[1] = { count };
            size_t localWorkSizes[1] = { workGroupSize };
            queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
            queue->finish();
        }

        void download(CommandQueue* queue, T* data)
        {
            queue->enqueueRead(out, data);
            queue->finish();
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

#endif // PARALLELBITONICSORTLOCAL_H
