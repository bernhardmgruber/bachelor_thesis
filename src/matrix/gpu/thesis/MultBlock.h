#pragma once

#include "../../../common/utils.h"
#include "../../../common/CLAlgorithm.h"
#include "../../MatrixAlgorithm.h"

namespace gpu
{
    namespace thesis
    {
        template<typename T>
        class MultBlock : public CLAlgorithm<T>, public MatrixAlgorithm
        {
            static_assert(is_same<T, float>::value, "Thesis algorithms only support float");

        public:
            static const size_t BLOCK_SIZE = 4;

            const string getName() override
            {
                return "Matrix multiplication (Blocks, THESIS dixxi AMD)";
            }

            const cl_uint getWorkDimensions() const override
            {
                return 2;
            }

            void init() override
            {
                Program* program = context->createProgram("gpu/thesis/MultBlock.cl");
                kernel = program->createKernel("MultBlocks");
                delete program;
            }

            void upload(size_t workGroupSize, T* data, size_t size) override
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

            void run(size_t workGroupSize, size_t size) override
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
                    queue->enqueueRead(c, result, 0);

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
