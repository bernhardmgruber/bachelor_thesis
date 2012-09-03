#ifndef PARALLELSELECTIONSORTLOCAL_H
#define PARALLELSELECTIONSORTLOCAL_H

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
        class ParallelSelectionSortLocal : public GPUSortingAlgorithm<T, count>
        {
                using Base = GPUSortingAlgorithm<T, count>;

            public:
                string getName() override
                {
                    return "Parallel selection local (Bealto)";
                }

                bool init(Context* context) override
                {
                    program = context->createProgram("gpu/bealto/ParallelSelectionSortLocal.cl");
                    kernel = program->createKernel("ParallelSelectionSortLocal");

                    return true;
                }

                void upload(Context* context, T* data) override
                {
                    in = context->createBuffer(CL_MEM_READ_ONLY, sizeof(T) * count, data);
                    out = context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * count);
                }

                void sort(CommandQueue* queue, size_t workGroupSize) override
                {
                    size_t globalWorkSizes[1] = { count };
                    size_t localWorkSizes[1] = { workGroupSize };

                    kernel->setArg(0, in);
                    kernel->setArg(1, out);
                    kernel->setArg(2, sizeof(cl_uint) * localWorkSizes[0], nullptr); // local memory
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
                Program* program;
                Kernel* kernel;
                Buffer* in;
                Buffer* out;
        };
    }
}

#endif // PARALLELSELECTIONSORTLOCAL_H
