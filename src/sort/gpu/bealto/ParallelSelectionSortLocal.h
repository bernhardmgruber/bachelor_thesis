#ifndef PARALLELSELECTIONSORTLOCAL_H
#define PARALLELSELECTIONSORTLOCAL_H

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
        template<typename T>
        class ParallelSelectionSortLocal : public GPUAlgorithm<T>, public SortAlgorithm
        {
            public:
                const string getName() override
                {
                    return "Parallel selection local (Bealto)";
                }

                bool isInPlace() override
                {
                    return false;
                }

                void init(Context* context) override
                {
                    program = context->createProgram("gpu/bealto/ParallelSelectionSortLocal.cl", "-D T=" + getTypeName<T>());
                    kernel = program->createKernel("ParallelSelectionSortLocal");
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
                {
                    in = context->createBuffer(CL_MEM_READ_ONLY, sizeof(T) * size);
                    queue->enqueueWrite(in, data);
                    out = context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * size);
                }

                void run(CommandQueue* queue, size_t workGroupSize, size_t size) override
                {
                    size_t globalWorkSizes[1] = { size };
                    size_t localWorkSizes[1] = { workGroupSize };

                    kernel->setArg(0, in);
                    kernel->setArg(1, out);
                    kernel->setArg(2, sizeof(cl_uint) * localWorkSizes[0], nullptr); // local memory
                    queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
                }

                void download(CommandQueue* queue, T* result, size_t size) override
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

                virtual ~ParallelSelectionSortLocal() {}

            private:
                Program* program;
                Kernel* kernel;
                Buffer* in;
                Buffer* out;
        };
    }
}

#endif // PARALLELSELECTIONSORTLOCAL_H
