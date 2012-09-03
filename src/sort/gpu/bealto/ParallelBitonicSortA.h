#ifndef PARALLELBITONICSORTA_H
#define PARALLELBITONICSORTA_H

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
        class ParallelBitonicSortA : public GPUSortingAlgorithm<T, count>
        {
            public:
                string getName() override
                {
                    return "Parallel bitonic A (Bealto)";
                }

                void init(Context* context) override
                {
                    program = context->createProgram("gpu/bealto/ParallelBitonicSortA.cl");
                    kernel = program->createKernel("ParallelBitonicSortA");
                }

                void upload(Context* context, T* data) override
                {
                    in = context->createBuffer(CL_MEM_READ_ONLY, sizeof(T) * count, data);
                    out = context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * count);
                }

                void sort(CommandQueue* queue, size_t workGroupSize) override
                {
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
                            queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
                            queue->enqueueBarrier();
                            swapBuffers = !swapBuffers;
                        }
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

#endif // PARALLELBITONICSORTA_H
