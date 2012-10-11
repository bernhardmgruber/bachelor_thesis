#ifndef GPUDIXXISCANTASK_H
#define GPUDIXXISCANTASK_H

#include "../../ScanAlgorithm.h"
#include "../../../common/GPUAlgorithm.h"

#include "clpp/clppScan_GPU.h"

namespace gpu
{
    namespace dixxi
    {
        template<typename T>
        class ScanTask : public GPUAlgorithm<T>, public ScanAlgorithm
        {
            public:
                const string getName() override
                {
                    return "Scan Task (dixxi)";
                }

                bool isInclusiv() override
                {
                    return true;
                }

                void init(Context* context) override
                {
                    Program* program = context->createProgram("gpu/dixxi/ScanTask.cl", "-D T=" + getTypeName<T>());
                    kernel = program->createKernel("ScanTask");
                    delete program;
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
                {
                    dataBuffer = context->createBuffer(CL_MEM_READ_ONLY, sizeof(T) * size);
                    queue->enqueueWrite(dataBuffer, data);
                    resultBuffer = context->createBuffer(CL_MEM_WRITE_ONLY, sizeof(T) * size);
                }

                void run(CommandQueue* queue, size_t workGroupSize, size_t size) override
                {
                    kernel->setArg(0, dataBuffer);
                    kernel->setArg(1, resultBuffer);
                    kernel->setArg(2, (cl_uint)size);
                    queue->enqueueTask(kernel);
                }

                void download(CommandQueue* queue, T* result, size_t size) override
                {
                    queue->enqueueRead(resultBuffer, result);
                    delete dataBuffer;
                    delete resultBuffer;
                }

                void cleanup() override
                {
                    delete kernel;
                }

                virtual ~ScanTask() {}

            private:
                Buffer* dataBuffer;
                Buffer* resultBuffer;
                Kernel* kernel;
        };
    }
}

#endif // GPUDIXXISCANTASK_H