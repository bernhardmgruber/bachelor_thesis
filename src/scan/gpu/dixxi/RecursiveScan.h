#pragma once

#include <vector>

#include "../../ScanAlgorithm.h"
#include "../../../common/CLAlgorithm.h"

#include "../../../common/utils.h"

namespace gpu
{
    namespace dixxi
    {
        /**
         * Idea from: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
         * Chapter: 39.2.3 Avoiding Bank Conflicts and 39.2.4 Arrays of Arbitrary Size
         */
        template<typename T>
        class RecursiveScan : public CLAlgorithm<T>, public ScanAlgorithm
        {
            public:
                /**
                 * Constructor.
                 *
                 * @param useOptimizedKernel: Uses a local memory access optimized kernel implementation when set to true. Address computation will be slower.
                 */
                RecursiveScan(bool useOptimizedKernel = false)
                    : useOptimizedKernel(useOptimizedKernel)
                {
                };

                const string getName() override
                {
                    return "Local Scan (GPU Gems) (exclusiv)" + string(useOptimizedKernel ? " (optimized)" : "");
                }

                bool isInclusiv() override
                {
                    return false;
                }

                void init() override
                {
                    Program* program = context->createProgram("gpu/dixxi/RecursiveScan.cl", "-D T=" + getTypeName<T>());
                    kernel = program->createKernel(useOptimizedKernel ? "LocalScanOptim" : "LocalScan");
                    addKernel = program->createKernel("AddSums");
                    delete program;
                }

                void upload(size_t workGroupSize, T* data, size_t size) override
                {
                    if(workGroupSize == 1)
                        throw OpenCLException("work group size must be greater than 1");

                    bufferSize = roundToMultiple(size, workGroupSize * 2);

                    // note: when using CL_MEM_COPY_HOST_PTR the data is copied into the context (not onto the device).
                    // The data may still be stored in the host memory and copied to the device on demand.
                    //Therefore, we use an explicit enquueeRead() to ensure the data to be on the device.
                    buffer = context->createBuffer(CL_MEM_READ_WRITE, bufferSize * sizeof(T));
                    queue->enqueueWrite(buffer, data);
                }

                void run(size_t workGroupSize, size_t size) override
                {
                    run_r(workGroupSize, buffer, size);
                }

                void run_r(size_t workGroupSize, Buffer* blocks, size_t size)
                {
                    Buffer* sums = context->createBuffer(CL_MEM_READ_WRITE, roundToMultiple(blocks->getSize() / workGroupSize, workGroupSize * 2 * sizeof(T)));

                    size_t globalWorkSizes[] = { blocks->getSize() / sizeof(T) / 2 }; // the global work size is the half number of elements (each thread processed 2 elements)
                    size_t localWorkSizes[] = { min(workGroupSize, globalWorkSizes[0]) };

                    //cout << "global " << globalWorkSizes[0] << " local " << localWorkSizes[0] << endl;

                    kernel->setArg(0, blocks);
                    kernel->setArg(1, sums);
                    kernel->setArg(2, sizeof(T) * 2 * localWorkSizes[0], nullptr);
                    queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);

                    if(blocks->getSize() / sizeof(T) > localWorkSizes[0] * 2)
                    {
                        // the buffer containes more than one scanned block, scan the created sum buffer
                        run_r(workGroupSize, sums, size);

                        // get the remaining available local memory
                        size_t totalGlobalWorkSize = blocks->getSize() / sizeof(T) / 2;
                        size_t maxLocalMemSize = context->getInfo<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE) / 8; // FIXME: This does not work for NVIDIA cards, raises OUT_OF_RESOURCES
                        size_t maxGlobalWorkSize = maxLocalMemSize / sizeof(cl_int) * workGroupSize;

                        size_t offset = 0;

                        do
                        {
                            globalWorkSizes[0] = min(totalGlobalWorkSize - offset, maxGlobalWorkSize);
                            localWorkSizes[0] = min(workGroupSize, globalWorkSizes[0]);
                            size_t globalWorkOffsets[] = { offset };

                            // apply the sums to the buffer
                            addKernel->setArg(0, blocks);
                            addKernel->setArg(1, sums);
                            addKernel->setArg(2, min(sums->getSize(), maxLocalMemSize), nullptr);
                            queue->enqueueKernel(addKernel, 1, globalWorkSizes, localWorkSizes, globalWorkOffsets);

                            offset += maxGlobalWorkSize;
                        }
                        while(offset < totalGlobalWorkSize);
                    }

                    delete sums;
                }

                void download(T* result, size_t size) override
                {
                    queue->enqueueRead(buffer, result, 0, size * sizeof(T));
                    delete buffer;
                }

                void cleanup() override
                {
                    delete kernel;
                    delete addKernel;
                }

                virtual ~RecursiveScan() {}

            private:
                size_t bufferSize;
                Kernel* kernel;
                Kernel* addKernel;
                Buffer* buffer;
                bool useOptimizedKernel;
        };
    }
}
