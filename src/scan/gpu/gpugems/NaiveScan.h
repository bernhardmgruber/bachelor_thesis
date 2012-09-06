#ifndef GPUGEMSNAIVESCAN_H
#define GPUGEMSNAIVESCAN_H

#include "../../GPUScanAlgorithm.h"

#include "../../../common/utils.h"

namespace gpu
{
    namespace gpugems
    {
        /**
         * From: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
         * Chapter: 39.2.1 A Naive Parallel Scan
         */
        template<typename T, size_t count>
        class NaiveScan : public GPUScanAlgorithm<T, count>
        {
            public:
                string getName() override
                {
                    return "Naive Scan (GPU Gems)";
                }

                void init(Context* context) override
                {
                    Program* program = context->createProgram("gpu/gpugems/NaiveScan.cl");
                    kernel = program->createKernel("NaiveScan");
                }

                void upload(Context* context, T* data) override
                {
                    bufferSize = pow2roundup(count);

                    source = context->createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bufferSize * sizeof(T), data);
                    destination = context->createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bufferSize * sizeof(T), data);
                }

                void scan(CommandQueue* queue, size_t workGroupSize) override
                {
                    for(size_t dpower = 1; dpower < bufferSize; dpower <<= 1)
                    {
                        kernel->setArg(0, source);
                        kernel->setArg(1, destination);
                        kernel->setArg(2, dpower);

                        size_t globalWorkSizes[] = { count };
                        size_t localWorkSizes[] = { workGroupSize };

                        queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
                        queue->enqueueCopy(destination, source);
                    }
                    queue->finish();
                }

                void download(CommandQueue* queue, T* result) override
                {
                    queue->enqueueRead(source, result, 0, count * sizeof(T));
                }

                void cleanup() override
                {
                    delete kernel;
                    delete source;
                    delete destination;
                }

                virtual ~NaiveScan() {}

            private:
                size_t bufferSize;
                Kernel* kernel;
                Buffer* source;
                Buffer* destination;
        };
    }
}


#endif // GPUGEMSNAIVESCAN_H
