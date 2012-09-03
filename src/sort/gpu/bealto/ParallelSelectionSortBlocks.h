#ifndef PARALLELSELECTIONSORTBLOCKS_H
#define PARALLELSELECTIONSORTBLOCKS_H

#include <sstream>

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
        class ParallelSelectionSortBlocks : public GPUSortingAlgorithm<T, count>
        {
            public:
                ParallelSelectionSortBlocks(Context* context, CommandQueue* queue)
                {
                    blockFactor = 1;
                }

                virtual ~ParallelSelectionSortBlocks()
                {
                }

                string getName() override
                {
                    return "Parallel selection sort blocks (Bealto)";
                }

                bool init(Context* context) override
                {
                    stringstream options;
                    options << "-D BLOCK_FACTOR=" << blockFactor;
                    program = context->createProgram("gpu/bealto/ParallelSelectionSortBlocks.cl", options.str());
                    kernel = program->createKernel("ParallelSelectionSortBlocks");

                    return true;
                }

                void upload(Context* context, T* data) override
                {
                    in = context->createBuffer(CL_MEM_READ_ONLY, sizeof(T) * count, data);
                    out = context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * count);
                }

                void sort(CommandQueue* queue, size_t workGroupSize) override
                {
                    kernel->setArg(0, in);
                    kernel->setArg(1, out);
                    kernel->setArg(2, sizeof(cl_uint) * workGroupSize * blockFactor, nullptr);
                    size_t globalWorkSizes[1] = { count };
                    size_t localWorkSizes[1] = { workGroupSize };
                    queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
                    queue->finish();
                }

                void download(CommandQueue* queue, T* data) override
                {
                    queue->enqueueRead(out, data);
                    queue->finish();
                }

                void cleanup() override
                {
                    delete program;
                    delete in;
                    delete out;
                    delete kernel;
                }

            private:
                size_t blockFactor;
                Program* program;
                Kernel* kernel;
                Buffer* in;
                Buffer* out;
        };
    }
}

#endif // PARALLELSELECTIONSORTBLOCKS_H
