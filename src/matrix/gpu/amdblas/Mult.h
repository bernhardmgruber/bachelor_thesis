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

                void init() override
                {
                    cl_int err = clAmdBlasSetup();
                    if (err != CL_SUCCESS)
                        throw OpenCLException("clAmdBlasSetup() failed with " + err);
                }

                void upload(size_t workGroupSize, T* data, size_t size) override
                {
                    size_t matrixSize = size * size * sizeof(T);

                    a = context->createBuffer(CL_MEM_READ_ONLY, matrixSize);
                    b = context->createBuffer(CL_MEM_READ_ONLY, matrixSize);
                    c = context->createBuffer(CL_MEM_READ_WRITE, matrixSize);

                    queue->enqueueWrite(a, data);
                    queue->enqueueWrite(b, data + size * size);
                }

                void run(size_t workGroupSize, size_t size) override
                {
                    cl_command_queue clQueue = queue->getCLCommandQueue();
                    cl_int err = clAmdBlasSgemm(clAmdBlasRowMajor, clAmdBlasNoTrans, clAmdBlasNoTrans, size, size, size, 1.0, a->getCLBuffer(), size, b->getCLBuffer(), size, 0.0, c->getCLBuffer(), size, 1, &clQueue, 0, nullptr, nullptr);
                    if (err != CL_SUCCESS)
                        throw OpenCLException("clAmdBlasSgemm() failed with " + err);
                }

                void download(T* result, size_t size) override
                {
                    queue->enqueueRead(c, result, 0, size * size * sizeof(T));
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
                Buffer* a;
                Buffer* b;
                Buffer* c;
        };
    }
}
