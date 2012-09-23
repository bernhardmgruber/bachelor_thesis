#ifndef GPUGEMSWORKEFFICIENTSCAN_H
#define GPUGEMSWORKEFFICIENTSCAN_H

#include "../../ScanAlgorithm.h"
#include "../../../common/GPUAlgorithm.h"

#include "../../../common/utils.h"

namespace gpu
{
    namespace gpugems
    {
        /**
         * From: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
         * Chapter: 39.2.2 A Work-Efficient Parallel Scan
         */
        template<typename T, size_t count>
        class WorkEfficientScan : public GPUAlgorithm<T, count>, public ScanAlgorithm
        {
            public:
                string getName() override
                {
                    return "Work Efficient Scan (GPU Gems) (exclusiv)";
                }

                bool isInclusiv() override
                {
                    return false;
                }

                void init(Context* context) override
                {
                    Program* program = context->createProgram("gpu/gpugems/WorkEfficientScan.cl", "-D T=" + getTypeName<T>());
                    upSweepKernel = program->createKernel("UpSweep");
                    setLastZeroKernel = program->createKernel("SetLastZeroSweep");
                    downSweepKernel = program->createKernel("DownSweep");
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data) override
                {
                    bufferSize = pow2roundup(count);

                    buffer = context->createBuffer(CL_MEM_READ_WRITE, bufferSize * sizeof(T));
                    queue->enqueueWrite(buffer, data);
                }

                void run(CommandQueue* queue, size_t workGroupSize) override
                {
                    size_t globalWorkSizes[] = { count };
                    size_t localWorkSizes[] = { workGroupSize };

                    // upsweep (reduce)
                    for(size_t offset = 1; offset < bufferSize; offset <<= 1)
                    {
                        size_t stride = 2 * offset;

                        upSweepKernel->setArg(0, buffer);
                        upSweepKernel->setArg(1, offset);
                        upSweepKernel->setArg(2, stride);

                        queue->enqueueKernel(upSweepKernel, 1, globalWorkSizes, localWorkSizes);
                        queue->enqueueBarrier();
                    }

                    // set last element to zero
                    setLastZeroKernel->setArg(0, buffer);
                    setLastZeroKernel->setArg(1, bufferSize - 1);
                    queue->enqueueTask(setLastZeroKernel);

                    // downsweep
                    for(size_t offset = bufferSize >> 1; offset >= 1; offset >>= 1)
                    {
                        size_t stride = 2 * offset;

                        downSweepKernel->setArg(0, buffer);
                        downSweepKernel->setArg(1, offset);
                        downSweepKernel->setArg(2, stride);

                        queue->enqueueKernel(downSweepKernel, 1, globalWorkSizes, localWorkSizes);
                        queue->enqueueBarrier();
                    }
                }

                void download(CommandQueue* queue, T* result) override
                {
                    queue->enqueueRead(buffer, result, 0, count * sizeof(T));
                }

                void cleanup() override
                {
                    delete upSweepKernel;
                    delete downSweepKernel;
                    delete setLastZeroKernel;
                    delete buffer;
                }

                virtual ~WorkEfficientScan() {}

            private:
                size_t bufferSize;
                Kernel* upSweepKernel;
                Kernel* downSweepKernel;
                Kernel* setLastZeroKernel;
                Buffer* buffer;
        };
    }
}


#endif // GPUGEMSWORKEFFICIENTSCAN_H
