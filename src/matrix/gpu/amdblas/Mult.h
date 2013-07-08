#pragma once

#include <clAmdBlas.h>
#include <stdlib.h>
#include "../../../common/utils.h"
#include "../../../common/CLAlgorithm.h"
#include "../../MatrixAlgorithm.h"

namespace gpu
{
    namespace amdblas
    {
        template<typename T>
        class Mult : public CLAlgorithm<T>, public MatrixAlgorithm
        {
        public:
            const string getName() override
            {
                return "Matrix multiplication (AMD BLAS)";
            }

            const cl_uint getWorkDimensions() const override
            {
                return 2;
            }

            void init() override
            {
                cl_int err = clAmdBlasSetup();
                if (err != CL_SUCCESS)
                    throw OpenCLException("clAmdBlasSetup() failed with " + err);
            }

            void upload(size_t workGroupSize, T* data, size_t size) override
            {
#if 0
                size_t matrixSize = size * size * sizeof(T);

                a = context->createBuffer(CL_MEM_READ_ONLY, matrixSize);
                b = context->createBuffer(CL_MEM_READ_ONLY, matrixSize);
                c = context->createBuffer(CL_MEM_READ_WRITE, matrixSize);

                queue->enqueueWrite(a, data);
                queue->enqueueWrite(b, data + size * size);
#else
                adjustedSize = roundToMultiple(size, workGroupSize);

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
#endif
            }

            void run(size_t workGroupSize, size_t size) override
            {
                cl_command_queue clQueue = queue->getCLCommandQueue();
                cl_int err = clAmdBlasSgemm(clAmdBlasRowMajor, clAmdBlasNoTrans, clAmdBlasNoTrans, adjustedSize, adjustedSize, adjustedSize, 1.0, a->getCLBuffer(), adjustedSize, b->getCLBuffer(), adjustedSize, 0.0, c->getCLBuffer(), adjustedSize, 1, &clQueue, 0, nullptr, nullptr);
                if (err != CL_SUCCESS)
                    throw OpenCLException("clAmdBlasSgemm() failed with " + err);
            }

            void download(T* result, size_t size) override
            {
#if 0
                queue->enqueueRead(c, result, 0, size * size * sizeof(T));
#else
                if(adjustedSize != size)
                {
                    size_t bufferOffset[] = {0, 0, 0};
                    size_t hostOffset[] = {0, 0, 0};
                    size_t sizes[] = {size * sizeof(T), size, 1};
                    queue->enqueueReadRect(c, result, bufferOffset, hostOffset, sizes, adjustedSize * sizeof(T), 0, size * sizeof(T), 0);
                }
                else
                    queue->enqueueRead(c, result, 0);
#endif
                delete a;
                delete b;
                delete c;
            }

            void cleanup() override
            {
                clAmdBlasTeardown();
            }

            virtual ~Mult() {}

        private:
            size_t adjustedSize;

            Buffer* a;
            Buffer* b;
            Buffer* c;
        };
    }
}
