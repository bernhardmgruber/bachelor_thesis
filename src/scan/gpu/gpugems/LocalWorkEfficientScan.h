#pragma once

#include "../../ScanAlgorithm.h"
#include "../../../common/CLAlgorithm.h"

#include "../../../common/utils.h"

namespace gpu
{
    namespace gpugems
    {
        /**
        * From: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
        * Chapter: 39.2.2 A Work-Efficient Parallel Scan
        * Scans only local blocks
        */
        template<typename T>
        class LocalWorkEfficientScan : public CLAlgorithm<T>, public ScanAlgorithm
        {
        public:
            const string getName() override
            {
                return "Work Efficient Scan (GPU Gems) (exclusiv)";
            }

            bool isInclusiv() override
            {
                return false;
            }

            void init() override
            {
                Program* program = context->createProgram("gpu/gpugems/LocalWorkEfficientScan.cl", "-D T=" + getTypeName<T>());
                kernel = program->createKernel("WorkEfficientScan");
                delete program;
            }

            void upload(size_t workGroupSize, T* data, size_t size) override
            {
                bufferSize = roundToPowerOfTwo(roundToMultiple(size / 2, workGroupSize));

                source = context->createBuffer(CL_MEM_READ_ONLY, bufferSize * sizeof(T));
                queue->enqueueWrite(source, data, 0, size * sizeof(T));
 
                destination = context->createBuffer(CL_MEM_WRITE_ONLY, bufferSize * sizeof(T));
            }

            void run(size_t workGroupSize, size_t size) override
            {
                kernel->setArg(0, source);
                kernel->setArg(1, destination);
                kernel->setArg(2, workGroupSize * sizeof(T) * 2, nullptr);

                size_t globalWorkSizes[] = { bufferSize / 2 };
                size_t localWorkSizes[] = { workGroupSize };

                queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
            }

            void download(T* result, size_t size) override
            {
                queue->enqueueRead(destination, result, 0, size * sizeof(T));
                printArr(result, size);
                delete source;
                delete destination;
            }

            void cleanup() override
            {
                delete kernel;
            }

            virtual ~LocalWorkEfficientScan() {}

        private:
            size_t bufferSize;
            Kernel* kernel;
            Buffer* source;
            Buffer* destination;
        };
    }
}
