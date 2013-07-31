#pragma once

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
        * Scans only local blocks
        */
        template<typename T>
        class LocalWorkEfficientVecScan : public CLAlgorithm<T>, public ScanAlgorithm
        {
            static const int BLOCK_SIZE = 4;

        public:
            const string getName() override
            {
                return "Work Efficient Vec Scan (dixxi GPU Gems) (exclusiv)";
            }

            bool isInclusiv() override
            {
                return false;
            }

            void init() override
            {
                stringstream ss;
                ss << "-D T=" << getTypeName<T>() << " -D BLOCK_SIZE=" << BLOCK_SIZE << " -D BLOCK_SIZE_MINUS_ONE=" << BLOCK_SIZE - 1;
                Program* program = context->createProgram("gpu/dixxi/LocalWorkEfficientVecScan.cl", ss.str());
                kernel = program->createKernel("WorkEfficientVecScan");
                delete program;
            }

            void upload(size_t workGroupSize, T* data, size_t size) override
            {
                bufferSize = roundToMultiple(size, workGroupSize * 2 * BLOCK_SIZE);

                buffer = context->createBuffer(CL_MEM_READ_WRITE, bufferSize * sizeof(T));
                queue->enqueueWrite(buffer, data, 0, size * sizeof(T));
            }

            void run(size_t workGroupSize, size_t size) override
            {
                kernel->setArg(0, buffer);
                kernel->setArg(1, workGroupSize * sizeof(T) * 2, nullptr);

                size_t globalWorkSizes[] = { bufferSize / (2 * BLOCK_SIZE) };
                size_t localWorkSizes[] = { workGroupSize };

                queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
            }

            void download(T* result, size_t size) override
            {
                queue->enqueueRead(buffer, result, 0, size * sizeof(T));
                delete buffer;
            }

            void cleanup() override
            {
                delete kernel;
            }

            virtual ~LocalWorkEfficientVecScan() {}

        private:
            size_t bufferSize;
            Kernel* kernel;
            Buffer* buffer;
        };
    }
}
