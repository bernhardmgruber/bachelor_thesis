#ifndef GPUAMDMULTTILE_H
#define GPUAMDMULTTILE_H

#include "../../../common/utils.h"
#include "../../../common/GPUAlgorithm.h"
#include "../../MatrixAlgorithm.h"

namespace gpu
{
    namespace amd
    {
        template<typename T>
        class MultTile : public GPUAlgorithm<T>, public MatrixAlgorithm
        {
            public:
                static const size_t BLOCK_SIZE = 4;

                const string getName() override
                {
                    return "Matrix multiplication (Tiles, AMD)";
                }

                void init(Context* context) override
                {
                    Program* program = context->createProgram("gpu/amd/MultTile.cl", "-D T4=" + getTypeName<T>() + "4");
                    kernel = program->createKernel("MultTile");
                    delete program;
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
                {
                    workGroupSize = 16;

                    adjustedSize = roundToMultiple(size, BLOCK_SIZE);

                    a = context->createBuffer(CL_MEM_READ_ONLY, adjustedSize * adjustedSize * sizeof(T));
                    if(adjustedSize != size)
                        queue->enqueueFill(a, (cl_float)0);
                    queue->enqueueWriteRect(a, data, (size_t[]){0, 0, 0}, (size_t[]){0, 0, 0}, (size_t[]){size * sizeof(T), size, 1}, adjustedSize * sizeof(T), 0, size * sizeof(T), 0);
                    //queue->enqueueWrite(a, data, size * size * sizeof(T));

                    b = context->createBuffer(CL_MEM_READ_ONLY, adjustedSize * adjustedSize * sizeof(T));
                    if(adjustedSize != size)
                        queue->enqueueFill(b, (cl_float)0);
                    queue->enqueueWriteRect(b, data + size * size, (size_t[]){0, 0, 0}, (size_t[]){0, 0, 0}, (size_t[]){size * sizeof(T), size, 1}, adjustedSize * sizeof(T), 0, size * sizeof(T), 0);
                    //queue->enqueueWrite(b, data, size * size * sizeof(T));

                    c = context->createBuffer(CL_MEM_WRITE_ONLY, adjustedSize * adjustedSize * sizeof(T));
                }

                void run(CommandQueue* queue, size_t workGroupSize, size_t size) override
                {
                    workGroupSize = 16;

                    kernel->setArg(0, a);
                    kernel->setArg(1, b);
                    kernel->setArg(2, c);
                    kernel->setArg(3, (cl_uint)adjustedSize);

                    size_t adjustedWorkSize = roundToMultiple(adjustedSize, workGroupSize * 4);

                    size_t globalWorkSizes[] = { adjustedWorkSize / BLOCK_SIZE, adjustedWorkSize / BLOCK_SIZE };
                    size_t localWorkSizes[] = { workGroupSize, workGroupSize };

                    queue->enqueueKernel(kernel, 2, globalWorkSizes, localWorkSizes);
                }

                void download(CommandQueue* queue, T* result, size_t size) override
                {
                    queue->enqueueReadRect(c, result, (size_t[]){0, 0, 0}, (size_t[]){0, 0, 0}, (size_t[]){size * sizeof(T), size, 1}, adjustedSize * sizeof(T), 0, size * sizeof(T), 0);
                    //queue->enqueueRead(c, result, 0, size * size * sizeof(T));

                    //printArr2D(result, size * size, size);
                }

                void cleanup() override
                {
                    delete kernel;
                    delete a;
                    delete b;
                    delete c;
                }

                virtual ~MultTile() {}

            private:
                Kernel* kernel;
                Buffer* a;
                Buffer* b;
                Buffer* c;
                size_t adjustedSize;
        };
    }
}

#endif // GPUAMDMULTTILE_H
