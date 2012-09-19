#ifndef GPUGEMSLOCALSCAN_H
#define GPUGEMSLOCALSCAN_H

#include <vector>

#include "../../ScanAlgorithm.h"
#include "../../../common/GPUAlgorithm.h"

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
        class LocalScan : public GPUAlgorithm<T, count>, public ScanAlgorithm
        {
            public:
                /**
                 * Constructor.
                 *
                 * @param useOptimizedKernel: Uses a local memory access optimized kernel implementation when set to true.
                 */
                LocalScan(bool useOptimizedKernel = false)
                    : useOptimizedKernel(useOptimizedKernel)
                {
                };

                string getName() override
                {
                    return "Local Scan (GPU Gems) (exclusiv)" + string(useOptimizedKernel ? " (optimized)" : "");
                }

                bool isInclusiv() override
                {
                    return false;
                }

                void init(Context* context) override
                {
                    Program* program = context->createProgram("gpu/gpugems/LocalScan.cl", "-D T=" + getTypeName<T>());
                    kernel = program->createKernel(useOptimizedKernel ? "LocalScanOptim" : "LocalScan");
                    addKernel = program->createKernel("AddSums");
                }

                void upload(Context* context, size_t workGroupSize, T* data) override
                {
                    if(workGroupSize == 1)
                        throw OpenCLException("work group size must be greater than 1");

                    bufferSize = roundToMultiple(count, workGroupSize * 2);

                    buffer = context->createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bufferSize * sizeof(T), data);
                }

                void run(CommandQueue* queue, size_t workGroupSize) override
                {
                    Context* context = queue->getContext();

                    run_r(context, queue, workGroupSize, buffer);

                    queue->finish();
                }

                void run_r(Context* context, CommandQueue* queue, size_t workGroupSize, Buffer* blocks)
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
                        run_r(context, queue, workGroupSize, sums);

                        // get the remaining available local memory
                        size_t totalGlobalWorkSize = blocks->getSize() / sizeof(T) / 2;
                        size_t maxLocalMemSize = context->getInfo<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE); // FIXME: This does not work for NVIDIA cards, raises OUT_OF_RESOURCES
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

                void download(CommandQueue* queue, T* result) override
                {
                    queue->enqueueRead(buffer, result, 0, count * sizeof(T));

                    //for(int i = 0; i < count; i++)
                    //    cout << result[i] << endl;
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
                bool useOptimizedKernel;
        };
    }
}


#endif // GPUGEMSLOCALSCAN_H
