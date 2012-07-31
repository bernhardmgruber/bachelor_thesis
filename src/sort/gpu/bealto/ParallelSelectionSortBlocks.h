#ifndef PARALLELSELECTIONSORTBLOCKS_H
#define PARALLELSELECTIONSORTBLOCKS_H

#include "../../GPUSortingAlgorithm.h"
#include "../../OpenCL.h"

using namespace std;

/**
 * From: http://www.bealto.com/gpu-sorting_intro.html
 */
template<typename T, size_t count>
class ParallelSelectionSortBlocks : public GPUSortingAlgorithm<T, count>
{
    using Base = GPUSortingAlgorithm<T, count>;

    public:
        ParallelSelectionSortBlocks(Context* context, CommandQueue* queue)
            : GPUSortingAlgorithm<T, count>("Parallel selection (Bealto)", context, queue)
        {
        }

        virtual ~ParallelSelectionSortBlocks()
        {
        }

    protected:
        bool init()
        {
            program = Base::context->createProgram("gpu/bealto/ParallelSelectionSortBlocks.cl");
            kernel = program->createKernel("ParallelSelectionSortBlocks");

            return true;
        }

        void upload()
        {
            in = Base::context->createBuffer(CL_MEM_READ_ONLY, sizeof(T) * count);
            out = Base::context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * count);

            Base::queue->enqueueWrite(in, SortingAlgorithm<T, count>::data);
            Base::queue->finish();
        }

        void sort()
        {
            kernel->setArg(0, in);
            kernel->setArg(1, out);
            size_t globalWorkSizes[1] = { count };
            size_t localWorkSizes[1] = { min(Base::context->getInfoSize(CL_DEVICE_MAX_WORK_GROUP_SIZE), count) };
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

#endif // PARALLELSELECTIONSORTBLOCKS_H
