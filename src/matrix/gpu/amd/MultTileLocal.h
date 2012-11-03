#ifndef GPUAMDMULTTILELOCAL_H
#define GPUAMDMULTTILELOCAL_H

#include "../../../common/utils.h"
#include "../../../common/GPUAlgorithm.h"
#include "../../MatrixAlgorithm.h"

namespace gpu
{
    namespace amd
    {
        template<typename T>
        class MultTileLocal : public GPUAlgorithm<T>, public MatrixAlgorithm
        {
            public:


                const string getName() override
                {
                    return "Matrix multiplication (Tiles local, AMD)";
                }

                void init(Context* context) override
                {
                    Program* program = context->createProgram("gpu/amd/MultTileLocal.cl", "-D T4=" + getTypeName<T>() + "4");
                    kernel = program->createKernel("MultTileLocal");
                    delete program;

                    size_t workGroupSize = kernel->getWorkGroupSize();

                    if(workGroupSize >= 64)
                    {
                        blockSize = 8;
                        localWorkSizes[0] = blockSize;
                        localWorkSizes[1] = blockSize;
                    }
                    else if(workGroupSize >= 32)
                    {
                        blockSize = 4;
                        localWorkSizes[0] = blockSize;
                        localWorkSizes[1] = blockSize;
                    }
                    else
                    {
                        stringstream ss;
                        ss << "Out of Resources! Group Size specified : " << localWorkSizes[0] * localWorkSizes[1] << ". Max Group Size supported on the kernel : " << workGroupSize << endl;
                        throw OpenCLException(ss.str());
                    }
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
                {
                    adjustedSize = roundToMultiple(size, blockSize);

                    a = context->createBuffer(CL_MEM_READ_ONLY, adjustedSize * adjustedSize * sizeof(T));
                    if(adjustedSize != size)
                        queue->enqueueFill(a, (cl_float)0);
                    queue->enqueueWriteRect(a, data, (size_t[])
                    {
                        0, 0, 0
                    }, (size_t[])
                    {
                        0, 0, 0
                    }, (size_t[])
                    {
                        size * sizeof(T), size, 1
                    }, adjustedSize * sizeof(T), 0, size * sizeof(T), 0);
                    //queue->enqueueWrite(a, data, size * size * sizeof(T));

                    b = context->createBuffer(CL_MEM_READ_ONLY, adjustedSize * adjustedSize * sizeof(T));
                    if(adjustedSize != size)
                        queue->enqueueFill(b, (cl_float)0);
                    queue->enqueueWriteRect(b, data + size * size, (size_t[])
                    {
                        0, 0, 0
                    }, (size_t[])
                    {
                        0, 0, 0
                    }, (size_t[])
                    {
                        size * sizeof(T), size, 1
                    }, adjustedSize * sizeof(T), 0, size * sizeof(T), 0);
                    //queue->enqueueWrite(b, data, size * size * sizeof(T));

                    c = context->createBuffer(CL_MEM_WRITE_ONLY, adjustedSize * adjustedSize * sizeof(T));
                }

                void run(CommandQueue* queue, size_t workGroupSize, size_t size) override
                {
                    kernel->setArg(0, a);
                    kernel->setArg(1, b);
                    kernel->setArg(2, c);
                    kernel->setArg(3, (cl_uint)adjustedSize);
                    kernel->setArg(4, (blockSize * 4) * (blockSize * 4) * sizeof(cl_float), nullptr);

                    size_t adjustedWorkSize = roundToMultiple(adjustedSize, localWorkSizes[0] * blockSize);

                    size_t globalWorkSizes[] = { adjustedWorkSize / 4, adjustedWorkSize / 4 };

                    queue->enqueueKernel(kernel, 2, globalWorkSizes, localWorkSizes);
                }

                void download(CommandQueue* queue, T* result, size_t size) override
                {
                    queue->enqueueReadRect(c, result, (size_t[])
                    {
                        0, 0, 0
                    }, (size_t[])
                    {
                        0, 0, 0
                    }, (size_t[])
                    {
                        size * sizeof(T), size, 1
                    }, adjustedSize * sizeof(T), 0, size * sizeof(T), 0);
                    //queue->enqueueRead(c, result, 0, size * size * sizeof(T));

                    printArr2D(result, size * size, size);
                }

                void cleanup() override
                {
                    delete kernel;
                    delete a;
                    delete b;
                    delete c;
                }

                virtual ~MultTileLocal() {}

            private:
                Kernel* kernel;
                Buffer* a;
                Buffer* b;
                Buffer* c;
                size_t adjustedSize;
                size_t blockSize;
                size_t localWorkSizes[2];
        };
    }
}

#endif // GPUAMDMULTTILELOCAL_H
