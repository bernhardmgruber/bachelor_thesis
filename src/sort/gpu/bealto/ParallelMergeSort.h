#ifndef PARALLELMERGESORT_H
#define PARALLELMERGESORT_H

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
        class ParallelMergeSort : public GPUSortingAlgorithm<T, count>
        {
                using Base = GPUSortingAlgorithm<T, count>;

            public:
                ParallelMergeSort(Context* context, CommandQueue* queue)
                    : GPUSortingAlgorithm<T, count>("Parallel merge (Bealto)", context, queue, true)
                {
                }

                virtual ~ParallelMergeSort()
                {
                }

            protected:
                void init(Context* context) override
                {
                    program = context->createProgram("gpu/bealto/ParallelMergeSort.cl");
                    kernel = program->createKernel("ParallelMergeSort");
                }

                void upload(Context* context, T* data) override
                {
                    in = Base::context->createBuffer(CL_MEM_READ_ONLY, sizeof(T) * count, data);
                    out = Base::context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * count);
                }

                void sort(CommandQueue* queue, size_t workGroupSize) override
                {
                    kernel->setArg(0, in);
                    kernel->setArg(1, out);
                    kernel->setArg(2, sizeof(cl_int) * workGroupSize, nullptr);
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
                Program* program;
                Kernel* kernel;
                Buffer* in;
                Buffer* out;
        };
    }
}

#endif // PARALLELMERGESORT_H
