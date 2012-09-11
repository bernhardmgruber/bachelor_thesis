#ifndef GPUGEMSLOCALSCAN_H
#define GPUGEMSLOCALSCAN_H

#include <vector>

#include "../../GPUScanAlgorithm.h"

#include "../../../common/utils.h"

namespace gpu
{
    namespace gpugems
    {
        /**
         * From: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
         * Chapter: 39.2.3 Avoiding Bank Conflicts and 39.2.4 Arrays of Arbitrary Size
         */
        template<typename T, size_t count>
        class LocalScan : public GPUScanAlgorithm<T, count>
        {
            public:
                string getName() override
                {
                    return "Local Scan (GPU Gems) (exclusiv)";
                }

                bool isInclusiv() override
                {
                    return false;
                }

                void init(Context* context) override
                {
                    Program* program = context->createProgram("gpu/gpugems/LocalScan.cl");
                    kernel = program->createKernel("LocalScan");
                    addKernel = program->createKernel("AddSums");
                }

                void upload(Context* context, size_t workGroupSize, T* data) override
                {
                    bufferSize = roundToMultiple(count, workGroupSize);

                    buffer = context->createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bufferSize * sizeof(T), data);
                }

                void scan(CommandQueue* queue, size_t workGroupSize) override
                {
                    Context* context = queue->getContext();

                    vector<Buffer*> buffers;
                    buffers.push_back(buffer);

                    do
                    {
                        Buffer* blocks = buffers.back();
                        Buffer* sums = context->createBuffer(CL_MEM_READ_WRITE, blocks->getSize() / workGroupSize);

                        size_t globalWorkSizes[] = { blocks->getSize() / sizeof(T) };
                        size_t localWorkSizes[] = { workGroupSize };

                        kernel->setArg(0, blocks);
                        kernel->setArg(1, sums);
                        kernel->setArg(2, sizeof(T) * 2 * workGroupSize, nullptr);
                        queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);

                        buffers.push_back(sums);
                    }
                    while(buffers.back()->getSize() / sizeof(T) > workGroupSize);

                    while(buffers.size() > 1)
                    {
                        Buffer* sums = buffers.back();
                        buffers.pop_back();
                        Buffer* blocks = buffers.back();

                        size_t globalWorkSizes[] = { blocks->getSize() / sizeof(T) };
                        size_t localWorkSizes[] = { workGroupSize };

                        addKernel->setArg(0, blocks);
                        addKernel->setArg(1, sums);
                        addKernel->setArg(2, sums->getSize(), nullptr);
                        //queue->enqueueKernel(addKernel, 1, globalWorkSizes, localWorkSizes);

                        delete sums;
                    }

                    queue->finish();
                }

                void download(CommandQueue* queue, T* result) override
                {
                    queue->enqueueRead(buffer, result, 0, count * sizeof(T));
                }

                void cleanup() override
                {
                    delete kernel;
                    delete addKernel;
                    delete buffer;
                }

                virtual ~LocalScan() {}

            private:
                size_t bufferSize;
                Kernel* kernel;
                Kernel* addKernel;
                Buffer* buffer;
        };
    }
}


#endif // GPUGEMSLOCALSCAN_H
