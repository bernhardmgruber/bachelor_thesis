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

                const cl_uint getWorkDimensions() const override
                {
                    return 2;
                }

                void init(Context* context) override
                {
                    stringstream ss;
                    ss << "-DT4=" << getTypeName<T>() << "4" << " -DBLOCK_SIZE=" << BLOCK_SIZE;
                    Program* program = context->createProgram("gpu/amd/MultTile.cl", ss.str());
                    kernel = program->createKernel("MultTile");
                    delete program;
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
                {
                    adjustedSize = roundToMultiple(size, BLOCK_SIZE);

                    a = context->createBuffer(CL_MEM_READ_ONLY, adjustedSize * adjustedSize * sizeof(T));
                    if(adjustedSize != size)
                    {
                        queue->enqueueFill(a, (cl_float)0);
                        size_t bufferOffset[] = {0, 0, 0};
                        size_t hostOffset[] = {0, 0, 0};
                        size_t sizes[] = {size * sizeof(T), size, 1};
                        queue->enqueueWriteRect(a, data, bufferOffset, hostOffset, sizes , adjustedSize * sizeof(T), 0, size * sizeof(T), 0);
                    }
                    else
                        queue->enqueueWrite(a, data);

                    b = context->createBuffer(CL_MEM_READ_ONLY, adjustedSize * adjustedSize * sizeof(T));
                    if(adjustedSize != size)
                    {
                        queue->enqueueFill(b, (cl_float)0);
                        size_t bufferOffset[] = {0, 0, 0};
                        size_t hostOffset[] = {0, 0, 0};
                        size_t sizes[] = {size * sizeof(T), size, 1};
                        queue->enqueueWriteRect(b, data + size * size, bufferOffset, hostOffset, sizes , adjustedSize * sizeof(T), 0, size * sizeof(T), 0);
                    }
                    else
                        queue->enqueueWrite(b, data + size * size);

                    c = context->createBuffer(CL_MEM_WRITE_ONLY, adjustedSize * adjustedSize * sizeof(T));
                }

                void run(CommandQueue* queue, size_t workGroupSize, size_t size) override
                {
                    kernel->setArg(0, a);
                    kernel->setArg(1, b);
                    kernel->setArg(2, c);
                    kernel->setArg(3, (cl_uint)adjustedSize);

                    size_t adjustedWorkSize = roundToMultiple(adjustedSize, workGroupSize * BLOCK_SIZE);

                    size_t globalWorkSizes[] = { adjustedWorkSize / BLOCK_SIZE, adjustedWorkSize / BLOCK_SIZE }; // each thread processes one block
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
                        queue->enqueueRead(c, result, 0);

                    delete a;
                    delete b;
                    delete c;
                }

                void cleanup() override
                {
                    delete kernel;
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
