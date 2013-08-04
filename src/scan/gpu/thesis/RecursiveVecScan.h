#pragma once

#include <vector>

#include "../../ScanAlgorithm.h"
#include "../../../common/CLAlgorithm.h"

#include "../../../common/utils.h"

namespace gpu
{
    namespace thesis
    {
        /**
        * Idea from: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
        * Chapter: 39.2.5 Further Optimization and Performance Results
        */
        template<typename T>
        class RecursiveVecScan : public CLAlgorithm<T>, public ScanAlgorithm
        {
            static_assert(is_same<T, cl_int>::value, "Thesis algorithms only support int");

            /// Uses a local memory access optimized kernel implementation when set to true. Address computation will be slower.
            static const bool AVOID_BANK_CONFLICTS = false;

            static const int VECTOR_WIDTH = 8;

        public:
            const string getName() override
            {
                return "Recursive Vec Scan (THESIS dixxi GPU Gems) (exclusiv)" + string(AVOID_BANK_CONFLICTS ? " (optimized)" : "");
            }

            const vector<size_t> getSupportedWorkGroupSizes() const override
            {
                // this algorithm does not allow a work group size of 1, because this would not reduce the problem size in a recursion.
                // work group sizes larger than 128 do not work for bank conflict avoidance
                auto sizes = CLAlgorithm<T>::getSupportedWorkGroupSizes();
                sizes.erase(remove_if(begin(sizes), end(sizes), [](size_t size) { return size < 2 || (size > 128 && AVOID_BANK_CONFLICTS); }), sizes.end());
                return sizes;
            }

            bool isInclusiv() override
            {
                return false;
            }

            void init() override
            {
                stringstream ss;
                Program* program = context->createProgram("gpu/thesis/RecursiveVecScan.cl");
                kernel = program->createKernel(AVOID_BANK_CONFLICTS ? "ScanBlocksVecOptim" : "ScanBlocksVec");
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

                kernel->setArg(0, values);
                kernel->setArg(1, sums);
                kernel->setArg(2, sizeof(T) * 2 * workGroupSize, nullptr);

                size_t globalWorkSizes[] = { size / (2 * VECTOR_WIDTH) }; // each thread processed 2 * VECTOR_WIDTH elements
                size_t localWorkSizes[] = { workGroupSize };

                queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);

                if(size > workGroupSize * 2 * VECTOR_WIDTH)
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
