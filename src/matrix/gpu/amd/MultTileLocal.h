#pragma once

#include "../../../common/utils.h"
#include "../../../common/CLAlgorithm.h"
#include "../../MatrixAlgorithm.h"

namespace gpu
{
    namespace amd
    {
        template<typename T>
        class MultTileLocal : public CLAlgorithm<T>, public MatrixAlgorithm
        {
        public:
            static const size_t BLOCK_SIZE = 4;

            const string getName() override
            {
                return "Matrix multiplication (Tiles local, AMD)";
            }

            const cl_uint getWorkDimensions() const override
            {
                return 2;
            }

            void init() override
            {
                    stringstream ss;
                    ss << "-DT4=" << getTypeName<T>() << "4" << " -DBLOCK_SIZE=" << BLOCK_SIZE;
                    Program* program = context->createProgram("gpu/amd/MultTileLocal.cl", ss.str());
                    kernel = program->createKernel("MultTileLocal");
                    delete program;
            }

            void upload(size_t workGroupSize, T* data, size_t size) override
            {
                tileSize = workGroupSize;

                if(tileSize < 4)
                    throw OpenCLException("Block size must be a larger than or equal to 4");

                cl_ulong localMemAvailable;
                clGetDeviceInfo(OpenCL::getGPUContext()->getCLDevice(), CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemAvailable, nullptr);

                size_t localMemRequired = tileSize * tileSize * BLOCK_SIZE * BLOCK_SIZE * sizeof(cl_float);
                if(localMemRequired > localMemAvailable) {
                    stringstream ss;
                    ss << "Required local memory " << localMemRequired << " for given tileSize of " << tileSize << " is larger than the available memory " << localMemAvailable << endl;
                    throw OpenCLException(ss.str());
                }

                adjustedSize = roundToMultiple(size, tileSize * 4);

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

            void run(size_t workGroupSize, size_t size) override
            {
                kernel->setArg(0, a);
                kernel->setArg(1, b);
                kernel->setArg(2, c);
                kernel->setArg(3, (cl_uint)adjustedSize);
                kernel->setArg(4, tileSize * tileSize * BLOCK_SIZE * BLOCK_SIZE * sizeof(cl_float), nullptr);

                size_t globalWorkSizes[] = { adjustedSize / 4, adjustedSize / 4 };
                size_t localWorkSizes[] = { tileSize, tileSize };

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
                    queue->enqueueRead(c, result, 0, size * size * sizeof(T));

                delete a;
                delete b;
                delete c;
            }

            void cleanup() override
            {
                delete kernel;
            }

            virtual ~MultTileLocal() {}

        private:
            Kernel* kernel;
            Buffer* a;
            Buffer* b;
            Buffer* c;
            size_t adjustedSize;
            size_t tileSize;
        };
    }
}
