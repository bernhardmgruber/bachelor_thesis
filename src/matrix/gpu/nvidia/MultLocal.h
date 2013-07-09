#pragma once

#include "../../../common/utils.h"
#include "../../../common/CLAlgorithm.h"
#include "../../MatrixAlgorithm.h"

#include <sstream>

using namespace std;

namespace gpu
{
    namespace nvidia
    {
        template<typename T>
        class MultLocal : public CLAlgorithm<T>, public MatrixAlgorithm
        {
            public:
                const static size_t TILE_SIZE = 16;

                const string getName() override
                {
                    return "Matrix multiplication (Local tiles, NVIDIA)";
                }

                const vector<size_t> getSupportedWorkGroupSizes() const override
                {
                    vector<size_t> sizes;
                    sizes.push_back(TILE_SIZE);
                    return sizes;
                }

                void init() override
                {
                    stringstream ss;
                    ss << "-D T=" << getTypeName<T>() << " -D TILE_SIZE=" << TILE_SIZE;

                    Program* program = context->createProgram("gpu/nvidia/MultLocal.cl", ss.str());
                    kernel = program->createKernel("matrixMul");
                    delete program;
                }

                void upload(size_t workGroupSize, T* data, size_t size) override
                {
                    adjustedSize = roundToMultiple(size, TILE_SIZE);

                    a = context->createBuffer(CL_MEM_READ_ONLY, adjustedSize * adjustedSize * sizeof(T));
                    if(adjustedSize != size)
                    {
                        queue->enqueueFill(a, (T)0);
                        size_t bufferOffset[] = {0, 0, 0};
                        size_t hostOffset[] = {0, 0, 0};
                        size_t sizes[] = {size * sizeof(T), size, 1};
                        queue->enqueueWriteRect(a, data, bufferOffset, hostOffset, sizes, adjustedSize * sizeof(T), 0, size * sizeof(T), 0);
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
                        queue->enqueueWriteRect(b, data + size * size, bufferOffset, hostOffset, sizes, adjustedSize * sizeof(T), 0, size * sizeof(T), 0);
                    }
                    else
                        queue->enqueueWrite(b, data + size * size);

                    c = context->createBuffer(CL_MEM_WRITE_ONLY, adjustedSize * adjustedSize * sizeof(T));
                }

                void run(size_t workGroupSize, size_t size) override
                {
                    kernel->setArg(0, c);
                    kernel->setArg(1, a);
                    kernel->setArg(2, b);
                    kernel->setArg(3, sizeof(T) * TILE_SIZE * TILE_SIZE, nullptr);
                    kernel->setArg(4, sizeof(T) * TILE_SIZE * TILE_SIZE, nullptr);
                    kernel->setArg(5, (cl_int)adjustedSize);

                    size_t globalWorkSizes[] = { adjustedSize, adjustedSize };
                    size_t localWorkSizes[] = { TILE_SIZE, TILE_SIZE };

                    queue->enqueueKernel(kernel, 2, globalWorkSizes, localWorkSizes);
                }

                void download(T* result, size_t size) override
                {
                    if(adjustedSize != size)
                    {
                        size_t bufferOffset[] = {0, 0, 0};
                        size_t hostOffset[] = {0, 0, 0};
                        size_t sizes[] = {size * sizeof(T), size, 1};
                        queue->enqueueReadRect(c, result, bufferOffset, hostOffset, sizes, adjustedSize * sizeof(T), 0, size * sizeof(T), 0);
                    }
                    else
                        queue->enqueueRead(c, result);

                    //printArr2D(result, size * size, size);
                    delete a;
                    delete b;
                    delete c;
                }

                void cleanup() override
                {
                    delete kernel;
                }

                virtual ~MultLocal() {}

            private:
                Kernel* kernel;
                Buffer* a;
                Buffer* b;
                Buffer* c;

                size_t adjustedSize;
        };
    }
}
