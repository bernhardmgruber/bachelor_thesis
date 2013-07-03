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
         * Chapter: 39.2.1 A Naive Parallel Scan
         */
        template<typename T>
        class NaiveScan : public CLAlgorithm<T>, public ScanAlgorithm
        {
            public:
                const string getName() override
                {
                    return "Naive Scan (GPU Gems) (inclusiv)";
                }

                bool isInclusiv() override
                {
                    return true;
                }

                void init() override
                {
                    Program* program = context->createProgram("gpu/gpugems/NaiveScan.cl", "-D T=" + getTypeName<T>());
                    kernel = program->createKernel("NaiveScan");
                    delete program;
                }

                void upload(size_t workGroupSize, T* data, size_t size) override
                {
                    bufferSize = pow2roundup(size);

                    source = context->createBuffer(CL_MEM_READ_WRITE, bufferSize * sizeof(T));
                    queue->enqueueWrite(source, data);

                    destination = context->createBuffer(CL_MEM_READ_WRITE, bufferSize * sizeof(T));
                    queue->enqueueCopy(source, destination);
                }

                void run(size_t workGroupSize, size_t size) override
                {
                    for(size_t dpower = 1; dpower < bufferSize; dpower <<= 1)
                    {
                        kernel->setArg(0, source);
                        kernel->setArg(1, destination);
                        kernel->setArg(2, (cl_uint)dpower);

                        size_t globalWorkSizes[] = { size };
                        size_t localWorkSizes[] = { workGroupSize };

                        queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
                        queue->enqueueCopy(destination, source);
                    }
                }

                void download(T* result, size_t size) override
                {
                    queue->enqueueRead(source, result, 0, size * sizeof(T));
                    delete source;
                    delete destination;
                }

                void cleanup() override
                {
                    delete kernel;
                }

                virtual ~NaiveScan() {}

            private:
                size_t bufferSize;
                Kernel* kernel;
                Buffer* source;
                Buffer* destination;
        };
    }
}
