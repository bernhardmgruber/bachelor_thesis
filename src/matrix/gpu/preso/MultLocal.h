#ifndef GPUPRESOMULTTILELOCAL_H
#define GPUPRESOMULTTILELOCAL_H

#include "../../../common/utils.h"
#include "../../../common/GPUAlgorithm.h"
#include "../../MatrixAlgorithm.h"

#include <sstream>

namespace gpu
{
    namespace preso
    {
        /**
        * From: http://www.cs.nyu.edu/~lerner/spring12/Preso07-OpenCL.pdf
        */
        template<typename T>
        class MultLocal : public GPUAlgorithm<T>, public MatrixAlgorithm
        {
            const static int BLOCK_SIZE = 16;

        public:
            const string getName() override
            {
                return "Matrix multiplication (Tiles local, Preso)";
            }

            const cl_uint getWorkDimensions() const override
            {
                return 2;
            }

            void init(Context* context) override
            {
                stringstream ss;
                ss << "-D T=" << getTypeName<T>() << " -D BLOCK_SIZE=" << BLOCK_SIZE;
                Program* program = context->createProgram("gpu/preso/MultTileLocal.cl", ss.str());
                kernel = program->createKernel("MultLocal");
                delete program;
            }

            void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
            {
                cl_ulong localMemAvailable;
                clGetDeviceInfo(OpenCL::getGPUContext()->getCLDevice(), CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemAvailable, nullptr);

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

                size_t globalWorkSizes[] = { adjustedSize, adjustedSize };
                size_t localWorkSizes[] = { BLOCK_SIZE, BLOCK_SIZE };

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

            virtual ~MultLocal() {}

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

#endif // GPUPRESOMULTTILELOCAL_H
