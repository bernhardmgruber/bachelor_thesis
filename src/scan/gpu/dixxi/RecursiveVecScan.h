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
        * Chapter: 39.2.5 Further Optimization and Performance Results
        */
        template<typename T>
        class RecursiveVecScan : public CLAlgorithm<T>, public ScanAlgorithm
        {
            static const int VECTOR_WIDTH = 8;

        public:
            /**
            * Constructor.
            *
            * @param useOptimizedKernel: Uses a local memory access optimized kernel implementation when set to true. Address computation will be slower.
            */
            RecursiveVecScan(bool useOptimizedKernel = false)
                : useOptimizedKernel(useOptimizedKernel)
            {
            };

            const string getName() override
            {
                return "Recursive Vec Scan (dixxi GPU Gems) (exclusiv)" + string(useOptimizedKernel ? " (optimized)" : "");
            }

            const vector<size_t> getSupportedWorkGroupSizes() const override
            {
                // this algorithm does not allow a work group size of 1, because this would not reduce the problem size in a recursion.
                // work group sizes larger than 128 do not work for bank conflict avoidance
                auto sizes = CLAlgorithm<T>::getSupportedWorkGroupSizes();
                sizes.erase(remove_if(begin(sizes), end(sizes), [](size_t size) { return size < 2 || (size > 128 && useOptimizedKernel); }), sizes.end());
                return sizes;
            }

            bool isInclusiv() override
            {
                return false;
            }

            void init() override
            {
                stringstream ss;
                ss << "-D T=" << getTypeName<T>() << " -D VECTOR_WIDTH=" << VECTOR_WIDTH << " -D VECTOR_WIDTH_MINUS_ONE_HEX=" << hex << VECTOR_WIDTH - 1;
                Program* program = context->createProgram("gpu/dixxi/RecursiveVecScan.cl", ss.str());
                kernel = program->createKernel(useOptimizedKernel ? "WorkEfficientVecScanOptim" : "WorkEfficientVecScan");
                addKernel = program->createKernel("AddSums");
                delete program;
            }

            void upload(size_t workGroupSize, T* data, size_t size) override
            {
                bufferSize = roundToMultiple(size, workGroupSize * 2 * VECTOR_WIDTH);

                buffer = context->createBuffer(CL_MEM_READ_WRITE, bufferSize * sizeof(T));
                queue->enqueueWrite(buffer, data, 0, size * sizeof(T));
            }

            void run(size_t workGroupSize, size_t size) override
            {
                run_r(workGroupSize, buffer, bufferSize);
            }

            void run_r(size_t workGroupSize, Buffer* values, size_t size)
            {
                size_t sumBufferSize = roundToMultiple(size / (workGroupSize * 2 * VECTOR_WIDTH), workGroupSize * 2 * VECTOR_WIDTH);

                Buffer* sums = context->createBuffer(CL_MEM_READ_WRITE, sumBufferSize * sizeof(T));

                size_t globalWorkSizes[] = { size / (2 * VECTOR_WIDTH) }; // each thread processed 2 * BLOCK_SIZE elements
                size_t localWorkSizes[] = { min(workGroupSize, globalWorkSizes[0]) };

                kernel->setArg(0, values);
                kernel->setArg(1, sums);
                kernel->setArg(2, sizeof(T) * 2 * localWorkSizes[0], nullptr);
                queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);

                if(size > localWorkSizes[0] * 2 * VECTOR_WIDTH)
                {
                    // the buffer containes more than one scanned block, scan the created sum buffer
                    run_r(workGroupSize, sums, sumBufferSize);

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

            virtual ~RecursiveVecScan() {}

        private:
            size_t bufferSize;
            Kernel* kernel;
            Kernel* addKernel;
            Buffer* buffer;
            bool useOptimizedKernel;
        };
    }
}
