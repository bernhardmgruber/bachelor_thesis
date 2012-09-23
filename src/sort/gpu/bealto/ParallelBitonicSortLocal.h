#ifndef PARALLELBITONICSORTLOCAL_H
#define PARALLELBITONICSORTLOCAL_H

#include "../../../common/GPUAlgorithm.h"
#include "../../SortAlgorithm.h"

using namespace std;

namespace gpu
{
    namespace bealto
    {
        /**
         * From: http://www.bealto.com/gpu-sorting_intro.html
         */
        template<typename T, size_t count>
        class ParallelBitonicSortLocal : public GPUAlgorithm<T, count>, public SortAlgorithm
        {
            public:
                string getName() override
                {
                    return "Parallel bitonic local (Bealto)";
                }

                bool isInPlace() override
                {
                    return false;
                }

                void init(Context* context) override
                {
                    program = context->createProgram("gpu/bealto/ParallelBitonicSortLocal.cl");
                    kernel = program->createKernel("ParallelBitonicSortLocal");
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data) override
                {
                    in = context->createBuffer(CL_MEM_READ_ONLY, sizeof(T) * count);
                    queue->enqueueWrite(in, data);
                    out = context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * count);
                }

                void run(CommandQueue* queue, size_t workGroupSize) override
                {
                    kernel->setArg(0, in);
                    kernel->setArg(1, out);
                    kernel->setArg(2, sizeof(cl_int) * workGroupSize, nullptr);
                    size_t globalWorkSizes[1] = { count };
                    size_t localWorkSizes[1] = { workGroupSize };
                    queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
                }

                void download(CommandQueue* queue, T* result) override
                {
                    queue->enqueueRead(out, result);
                }

                void cleanup() override
                {
                    delete program;
                    delete in;
                    delete out;
                    delete kernel;
                }

                virtual ~ParallelBitonicSortLocal() {}

            private:
                Program* program;
                Kernel* kernel;
                Buffer* in;
                Buffer* out;
        };
    }
}

#endif // PARALLELBITONICSORTLOCAL_H
