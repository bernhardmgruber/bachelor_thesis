#ifndef PARALLELSELECTIONSORT_H
#define PARALLELSELECTIONSORT_H

#include "../../GPUSortingAlgorithm.h"
#include "../../OpenCL.h"

using namespace std;

/**
 * From: http://www.bealto.com/gpu-sorting_intro.html
 */
template<typename T, size_t count>
class ParallelSelectionSort : public GPUSortingAlgorithm<T, count>
{
    using Base = GPUSortingAlgorithm<T, count>;

    public:
        ParallelSelectionSort(Context* context, CommandQueue* queue)
            : GPUSortingAlgorithm<T, count>("Parallel selection (Bealto)", context, queue)
        {
        }

        virtual ~ParallelSelectionSort()
        {
        }

    protected:
        bool init()
        {
            program = Base::context->createProgram("gpu/bealto/ParallelSelection.cl");
            kernel = program->createKernel("ParallelSelection");

            return true;
        }

        void upload()
        {
            in = Base::context->createBuffer(CL_MEM_READ_ONLY, sizeof(T) * count);
            out = Base::context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * count);

            queue->enqueueWrite(in, SortingAlgorithm<T, count>::data);

            queue->finish();
        }

        void sort()
        {
            //int wg = min(n,256);
            kernel->setArg(0, in);
            kernel->setArg(1, out);
            queue->enqueueKernel(kernel, 1);
            //c->enqueueKernel(targetDevice,kid,n,1,wg,1,EventVector());

            queue->finish();
        }

        void download()
        {
            queue->enqueueRead(out, SortingAlgorithm<T, count>::data);
            queue->finish();
        }

        void cleanup()
        {
            delete program;
            delete in;
            delete out;
            delete kernel;
            OpenCL::cleanup();
        }

        CommandQueue* queue;
        Program* program;
        Kernel* kernel;
        Buffer* in;
        Buffer* out;
};

#endif // PARALLELSELECTIONSORT_H
