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
                    bufferSize = roundToMultiple(count, workGroupSize * 2);

                    buffer = context->createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bufferSize * sizeof(T), data);
                }

                void scan(CommandQueue* queue, size_t workGroupSize) override
                {
                    Context* context = queue->getContext();

                    scan_r(context, queue, workGroupSize, buffer);

                    queue->finish();
                }

                void scan_r(Context* context, CommandQueue* queue, size_t workGroupSize, Buffer* blocks)
                {
                    Buffer* sums = context->createBuffer(CL_MEM_READ_WRITE, roundToMultiple(blocks->getSize() / workGroupSize, workGroupSize * 2 * sizeof(T)));

                    size_t globalWorkSizes[] = { blocks->getSize() / sizeof(T) / 2 }; // the global work size is the half number of elements (each thread processed 2 elements)
                    size_t localWorkSizes[] = { min(workGroupSize, globalWorkSizes[0]) };

                    cout << "global " << globalWorkSizes[0] << " local " << localWorkSizes[0] << endl;

                    kernel->setArg(0, blocks);
                    kernel->setArg(1, sums);
                    kernel->setArg(2, sizeof(T) * 2 * localWorkSizes[0], nullptr);
                    queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);

                    if(blocks->getSize() / sizeof(T) > localWorkSizes[0] * 2)
                    {
                        // the buffer containes more than one scanned block
                        scan_r(context, queue, workGroupSize, sums);

                        globalWorkSizes[0] = blocks->getSize() / sizeof(T) / 2;
                        localWorkSizes[0] = min(workGroupSize, globalWorkSizes[0]);

                        //size_t maxLocalMemSize = context->getInfoSize(CL_MAX_M)

                        addKernel->setArg(0, blocks);
                        addKernel->setArg(1, sums);
                        addKernel->setArg(2, sums->getSize(), nullptr);
                        queue->enqueueKernel(addKernel, 1, globalWorkSizes, localWorkSizes);
                    }

                    delete sums;
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
