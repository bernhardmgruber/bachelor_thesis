#pragma once

#include "../../../common/utils.h"
#include "../../../common/CLAlgorithm.h"
#include "../../MatrixAlgorithm.h"

namespace gpu
{
    namespace dixxi
    {
        template<typename T>
        class Mult1D : public CLAlgorithm<T>, public MatrixAlgorithm
        {
            public:
                const string getName() override
                {
                    return "Matrix multiplication 1D";
                }

                void init() override
                {
                    Program* program = context->createProgram("gpu/dixxi/Mult1D.cl", "-D T=" + getTypeName<T>());
                    kernel = program->createKernel("Mult");
                    delete program;
                }

                void upload(size_t workGroupSize, T* data, size_t size) override
                {
                    size_t elementCount = size * size;

                    a = context->createBuffer(CL_MEM_READ_ONLY, elementCount * sizeof(T));
                    queue->enqueueWrite(a, data, 0, elementCount * sizeof(T), false);

                    b = context->createBuffer(CL_MEM_READ_ONLY, elementCount * sizeof(T));
                    queue->enqueueWrite(b, data + elementCount, 0, elementCount * sizeof(T), false);

                    c = context->createBuffer(CL_MEM_WRITE_ONLY, elementCount * sizeof(T));
                }

                void run(size_t workGroupSize, size_t size) override
                {
                    kernel->setArg(0, a);
                    kernel->setArg(1, b);
                    kernel->setArg(2, c);
                    kernel->setArg(3, (cl_uint)size);

                    size_t globalWorkSizes[] = { roundToMultiple(size * size, workGroupSize) };
                    size_t localWorkSizes[] = { workGroupSize };

                    queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
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
                    delete kernel;
                }

                virtual ~Mult1D() {}

            private:
                Kernel* kernel;
                Buffer* a;
                Buffer* b;
                Buffer* c;
        };
    }
}
