#pragma once

#include "../../ScanAlgorithm.h"
#include "../../../common/CLAlgorithm.h"
#include "../../../common/utils.h"

namespace gpu
{
    namespace dixxi
    {
        /**
        * Full array version of: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
        * Chapter: 39.2.1 A Naive Parallel Scan
        */
        template<typename T>
        class NaiveScan : public CLAlgorithm<T>, public ScanAlgorithm
        {
        public:
            const string getName() override
            {
                return "Naive Scan (dixxi GPU Gems) (inclusiv)";
            }

            bool isInclusiv() override
            {
                return true;
            }

            void init() override
            {
                Program* program = context->createProgram("gpu/dixxi/NaiveScan.cl", "-D T=" + getTypeName<T>());
                kernel = program->createKernel("NaiveScan");
                delete program;
            }

            void upload(size_t workGroupSize, T* data, size_t size) override
            {
                source = context->createBuffer(CL_MEM_READ_WRITE, size * sizeof(T));
                queue->enqueueWrite(source, data);

                destination = context->createBuffer(CL_MEM_READ_WRITE, size * sizeof(T));
            }

            void run(size_t workGroupSize, size_t size) override
            {
                size_t adjustedSize = roundToMultiple(size, workGroupSize);

                for(size_t power = 1; power < size; power <<= 1)
                {
                    kernel->setArg(0, source);
                    kernel->setArg(1, destination);
                    kernel->setArg(2, (cl_uint)power);
                    kernel->setArg(3, (cl_uint)size);

                    size_t globalWorkSizes[] = { adjustedSize };
                    size_t localWorkSizes[] = { workGroupSize };

                    queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);

                    swap(destination, source);
                }
            }

            void download(T* result, size_t size) override
            {
                queue->enqueueRead(source, result);
                delete source;
                delete destination;
            }

            void cleanup() override
            {
                delete kernel;
            }

            virtual ~NaiveScan() {}

        private:
            Kernel* kernel;
            Buffer* source;
            Buffer* destination;
        };
    }
}
