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
                return "Recursive Scan (GPU Gems) (exclusiv)" + string(useOptimizedKernel ? " (optimized)" : "");
            }

            const vector<size_t> getSupportedWorkGroupSizes() const override
            {
                // this algorithm does not allow a work group size of 1, because this would not reduce the problem size in a recursion.
                auto sizes = CLAlgorithm<T>::getSupportedWorkGroupSizes();
                sizes.erase(remove_if(begin(sizes), end(sizes), [](size_t size) { return size < 2; }), sizes.end());
                return sizes;
            }

            bool isInclusiv() override
            {
                return false;
            }

            void init() override
            {
                Program* program = context->createProgram("gpu/dixxi/RecursiveScan.cl", "-D T=" + getTypeName<T>());
                kernel = program->createKernel(useOptimizedKernel ? "WorkEfficientScanOptim" : "WorkEfficientScan");
                addKernel = program->createKernel("AddSums");
                delete program;
            }

            void upload(size_t workGroupSize, T* data, size_t size) override
            {
                bufferSize = roundToMultiple(size, workGroupSize * 2);

                buffer = context->createBuffer(CL_MEM_READ_WRITE, bufferSize * sizeof(T));
                queue->enqueueWrite(buffer, data, 0, size * sizeof(T));
            }

            void run(size_t workGroupSize, size_t size) override
            {
                run_r(workGroupSize, buffer, size);
            }

            void run_r(size_t workGroupSize, Buffer* values, size_t size)
            {
                Buffer* sums = context->createBuffer(CL_MEM_READ_WRITE, roundToMultiple(values->getSize() / workGroupSize, workGroupSize * 2 * sizeof(T)));

                size_t globalWorkSizes[] = { values->getSize() / sizeof(T) / 2 }; // the global work size is the half number of elements (each thread processed 2 elements)
                size_t localWorkSizes[] = { min(workGroupSize, globalWorkSizes[0]) };

                kernel->setArg(0, values);
                kernel->setArg(1, sums);
                kernel->setArg(2, sizeof(T) * 2 * localWorkSizes[0], nullptr);
                queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);

                if(values->getSize() / sizeof(T) > localWorkSizes[0] * 2)
                {
                    // the buffer containes more than one scanned block, scan the created sum buffer
                    run_r(workGroupSize, sums, size);

                    // apply the sums to the buffer
                    addKernel->setArg(0, values);
                    addKernel->setArg(1, sums);
                    queue->enqueueKernel(addKernel, 1, globalWorkSizes, localWorkSizes);
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
