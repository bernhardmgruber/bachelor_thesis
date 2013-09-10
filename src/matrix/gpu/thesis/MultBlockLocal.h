#pragma once

#include "../../../common/utils.h"
#include "../../../common/CLAlgorithm.h"
#include "../../MatrixAlgorithm.h"

namespace gpu
{
    namespace thesis
    {
        template<typename T>
        class MultBlockLocal : public CLAlgorithm<T>, public MatrixAlgorithm
        {
            static_assert(is_same<T, float>::value, "Thesis algorithms only support float");

        public:
            static const size_t BLOCK_SIZE = 4;
            const static size_t TILE_SIZE = 16;

            const string getName() override
            {
                return "Matrix multiplication (Blocks and local tiles, THESIS dixxi AMD)";
            }

            const cl_uint getWorkDimensions() const override
            {
                return 2; 
            }

            const vector<size_t> getSupportedWorkGroupSizes() const override
            {
                vector<size_t> sizes;
                sizes.push_back(TILE_SIZE);
                return sizes;
            }

            void init() override
            {
                Program* program = context->createProgram("gpu/thesis/MultBlockLocal.cl");
                kernel = program->createKernel("BlocksAndTilesGPU");
                delete program;
            }

            void upload(size_t workGroupSize, T* data, size_t size) override
            {
                cl_ulong localMemAvailable;
                clGetDeviceInfo(OpenCL::getGPUContext()->getCLDevice(), CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemAvailable, nullptr);

                size_t localMemRequired = TILE_SIZE * TILE_SIZE * BLOCK_SIZE * BLOCK_SIZE * sizeof(cl_float) * 2;
                if(localMemRequired > localMemAvailable) {
                    stringstream ss;
                    ss << "Required local memory " << localMemRequired << " for given tileSize of " << TILE_SIZE << " is larger than the available memory " << localMemAvailable << endl;
                    throw OpenCLException(ss.str());
                }

                adjustedSize = roundToMultiple(size, TILE_SIZE * BLOCK_SIZE);

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

                size_t globalWorkSizes[] = { adjustedSize / BLOCK_SIZE, adjustedSize / BLOCK_SIZE };
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
                    queue->enqueueRead(c, result, 0, size * size * sizeof(T));

                delete a;
                delete b;
                delete c;
            }

            void cleanup() override
            {
                delete kernel;
            }

            virtual ~MultBlockLocal() {}

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
