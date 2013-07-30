#pragma once

#include "../../ScanAlgorithm.h"
#include "../../../common/CLAlgorithm.h"

#include "../../../common/utils.h"

namespace gpu
{
    namespace dixxi
    {
        /**
        * Adapted from: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
        * Chapter: 39.2.2 A Work-Efficient Parallel Scan
        * Scans only local blocks
        */
        template<typename T>
        class LocalWorkEfficientBlockScan : public CLAlgorithm<T>, public ScanAlgorithm
        {
            static const int BLOCK_SIZE = 4;

        public:
            const string getName() override
            {
                return "Work Efficient Block Scan (dixxi GPU Gems) (exclusiv)";
            }

            bool isInclusiv() override
            {
                return false;
            }

            void init() override
            {
                stringstream ss;
                ss << "-D T=" << getTypeName<T>() << " -D BLOCK_SIZE=" << BLOCK_SIZE << " -D BLOCK_SIZE_MINUS_ONE=" << BLOCK_SIZE - 1;
                Program* program = context->createProgram("gpu/dixxi/LocalWorkEfficientBlockScan.cl", ss.str());
                kernel = program->createKernel("WorkEfficientBlockScan");
                delete program;
            }

            void upload(size_t workGroupSize, T* data, size_t size) override
            {
                bufferSize = roundToPowerOfTwo(roundToMultiple(size, workGroupSize * (2 * BLOCK_SIZE)));

                source = context->createBuffer(CL_MEM_READ_ONLY, bufferSize * sizeof(T));
                queue->enqueueWrite(source, data, 0, size * sizeof(T));

                destination = context->createBuffer(CL_MEM_WRITE_ONLY, bufferSize * sizeof(T));
            }

            void run(size_t workGroupSize, size_t size) override
            {
                kernel->setArg(0, source);
                kernel->setArg(1, destination);
                kernel->setArg(2, workGroupSize * sizeof(T) * 2, nullptr);

                size_t globalWorkSizes[] = { bufferSize / (2 * BLOCK_SIZE) };
                size_t localWorkSizes[] = { workGroupSize };

                queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
            }

            void download(T* result, size_t size) override
            {
                queue->enqueueRead(destination, result, 0, size * sizeof(T));
                delete source;
                delete destination;
            }

            void cleanup() override
            {
                delete kernel;
            }

            virtual ~LocalWorkEfficientBlockScan() {}

        private:
            size_t bufferSize;
            Kernel* kernel;
            Buffer* source;
            Buffer* destination;
        };
    }
}
