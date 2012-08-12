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
        ParallelBitonicSortLocal(Context* context, CommandQueue* queue)
            : GPUSortingAlgorithm<T, count>("Parallel bitonic local (Bealto)", context, queue, true)
        {
        }

        virtual ~ParallelBitonicSortLocal()
        {
        }

    protected:
        bool init()
        {
            program = Base::context->createProgram("gpu/bealto/ParallelBitonicSortLocal.cl");
            kernel = program->createKernel("ParallelBitonicSortLocal");

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
            kernel->setArg(0, in);
            kernel->setArg(1, out);
            kernel->setArg(2, sizeof(cl_int) * workGroupSize, nullptr);
            size_t globalWorkSizes[1] = { count };
            size_t localWorkSizes[1] = { workGroupSize };
            Base::queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
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

#endif // PARALLELBITONICSORTLOCAL_H
