#ifndef GPUDIXXIMULTBLOCK_H
#define GPUDIXXIMULTBLOCK_H

#include "../../../common/utils.h"
#include "../../../common/GPUAlgorithm.h"
#include "../../MatrixAlgorithm.h"

namespace gpu
{
    namespace dixxi
    {
        template<typename T>
        class MultBlock : public GPUAlgorithm<T>, public MatrixAlgorithm
        {
            public:
                const string getName() override
                {
                    return "Matrix multiplication (blocked)";
                }

                const cl_uint getWorkDimensions() const override
                {
                    return 2;
                }

                void init(Context* context) override
                {
                    Program* program = context->createProgram("gpu/dixxi/MultBlock.cl", "-D T=" + getTypeName<T>());
                    kernel = program->createKernel("MultBlock");
                    delete program;
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
                {
                    if(workGroupSize > 8)
                        workGroupSize = 8;

                    adjustedSize = roundToMultiple(size, workGroupSize);

                    a = context->createBuffer(CL_MEM_READ_ONLY, adjustedSize * adjustedSize * sizeof(T));
                    if(adjustedSize != size)
                    {
                        queue->enqueueFill(a, (cl_float)0);
                        size_t bufferOffset[] = {0, 0, 0};
                        size_t hostOffset[] = {0, 0, 0};
                        size_t sizes[] = {size * sizeof(T), size, 1};
                        queue->enqueueWriteRect(a, data, bufferOffset, hostOffset, sizes , adjustedSize * sizeof(T), 0, size * sizeof(T), 0, false);
                    }
                    else
                        queue->enqueueWrite(a, data, 0, size * size * sizeof(T), false);


                    b = context->createBuffer(CL_MEM_READ_ONLY, adjustedSize * adjustedSize * sizeof(T));
                    if(adjustedSize != size)
                    {
                        queue->enqueueFill(b, (cl_float)0);
                        size_t bufferOffset[] = {0, 0, 0};
                        size_t hostOffset[] = {0, 0, 0};
                        size_t sizes[] = {size * sizeof(T), size, 1};
                        queue->enqueueWriteRect(b, data + size * size, bufferOffset, hostOffset, sizes , adjustedSize * sizeof(T), 0, size * sizeof(T), 0, false);
                    }
                    else
                        queue->enqueueWrite(b, data, 0, size * size * sizeof(T), false);

                    c = context->createBuffer(CL_MEM_WRITE_ONLY, adjustedSize * adjustedSize * sizeof(T));
                }

                void run(CommandQueue* queue, size_t workGroupSize, size_t size) override
                {
                    if(workGroupSize > 8)
                        workGroupSize = 8;

                    kernel->setArg(0, a);
                    kernel->setArg(1, b);
                    kernel->setArg(2, c);
                    kernel->setArg(3, (cl_uint)size);
                    kernel->setArg(4, workGroupSize * workGroupSize * sizeof(cl_float), nullptr);
                    kernel->setArg(5, workGroupSize * workGroupSize * sizeof(cl_float), nullptr);

                    size_t globalWorkSizes[] = { adjustedSize, adjustedSize };
                    size_t localWorkSizes[] = { workGroupSize, workGroupSize };

                    queue->enqueueKernel(kernel, 2, globalWorkSizes, localWorkSizes);
                }

                void download(CommandQueue* queue, T* result, size_t size) override
                {
                    if(adjustedSize != size)
                    {
                        size_t bufferOffset[] = {0, 0, 0};
                        size_t hostOffset[] = {0, 0, 0};
                        size_t sizes[] = {size * sizeof(T), size, 1};
                        queue->enqueueReadRect(c, result, bufferOffset, hostOffset, sizes, adjustedSize * sizeof(T), 0, size * sizeof(T), 0);
                    }
                    else
                        queue->enqueueRead(c, result, 0, size * size * sizeof(T));
					delete a;
                    delete b;
                    delete c;
                }

                void cleanup() override
                {
                    delete kernel;
                }

                virtual ~MultBlock() {}

            private:
                Kernel* kernel;
                Buffer* a;
                Buffer* b;
                Buffer* c;
                size_t adjustedSize;
        };
    }
}

#endif // GPUDIXXIMULT_H
