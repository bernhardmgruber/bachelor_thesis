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

            const cl_uint getWorkDimensions() const override
            {
                return 2;
            }

            void init(Context* context) override
            {
                Program* program = context->createProgram("gpu/amd/MultTileLocal.cl", "-D T4=" + getTypeName<T>() + "4");
                kernel = program->createKernel("MultTileLocal");
                delete program;
            }

            void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
            {
                blockSize = workGroupSize = 16;

                if(blockSize < 4)
                    throw OpenCLException("Block size must be a larger than or equal to 4");

                cl_ulong localMemAvailable;
                clGetDeviceInfo(OpenCL::getGPUContext()->getCLDevice(), CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemAvailable, nullptr);

                size_t localMemRequired = (blockSize * 4) * (blockSize * 4) * sizeof(cl_float);
                if(localMemRequired > localMemAvailable) {
                    stringstream ss;
                    ss << "Required local memory " << localMemRequired << " for given blockSize of " << blockSize << " is larger than the available memory " << localMemAvailable << endl;
                    throw OpenCLException(ss.str());
                }

                adjustedSize = roundToMultiple(size, blockSize * 4);

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
                kernel->setArg(4, (blockSize * 4) * (blockSize * 4) * sizeof(cl_float), nullptr);

                size_t globalWorkSizes[] = { adjustedSize / 4, adjustedSize / 4 };
                size_t localWorkSizes[] = { blockSize, blockSize };

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

            virtual ~MultTileLocal() {}

        private:
            Kernel* kernel;
            Buffer* a;
            Buffer* b;
            Buffer* c;
            size_t adjustedSize;
            size_t blockSize;
        };
    }
}

#endif // GPUAMDMULTTILELOCAL_H
