#ifndef PARALLELBITONICSORTA_H
#define PARALLELBITONICSORTA_H

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
        class ParallelBitonicSortA : public GPUAlgorithm<T>, public SortAlgorithm
        {
            public:
                const string getName() override
                {
                    return "Parallel bitonic A (Bealto)";
                }

                bool isInPlace() override
                {
                    return false;
                }

                void init(Context* context) override
                {
                    Program* program = context->createProgram("gpu/bealto/ParallelBitonicSortA.cl", "-D T=" + getTypeName<T>());
                    kernel = program->createKernel("ParallelBitonicSortA");
                    delete program;
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
                {
                    in = context->createBuffer(CL_MEM_READ_ONLY, sizeof(T) * size);
                    queue->enqueueWrite(in, data);
                    out = context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * size);
                }

                void run(CommandQueue* queue, size_t workGroupSize, size_t size) override
                {
                    bool swapBuffers = false;

                    for (size_t length = 1; length < size; length <<= 1)
                        for (size_t inc = length; inc > 0; inc >>= 1)
                        {
                            kernel->setArg(0, swapBuffers ? out : in);
                            kernel->setArg(1, swapBuffers ? in : out);
                            kernel->setArg(2, inc);
                            kernel->setArg(3, length<<1);
                            size_t globalWorkSizes[1] = { size };
                            size_t localWorkSizes[1] = { workGroupSize };
                            queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
                            queue->enqueueBarrier();
                            swapBuffers = !swapBuffers;
                        }
                }

                void download(CommandQueue* queue, T* result, size_t size) override
                {
                    queue->enqueueRead(out, result);
                    delete in;
                    delete out;
                }

                void cleanup() override
                {
                    delete kernel;
                }

                virtual ~ParallelBitonicSortA() {}

            private:
                Kernel* kernel;
                Buffer* in;
                Buffer* out;
        };
    }
}

#endif // PARALLELBITONICSORTA_H
