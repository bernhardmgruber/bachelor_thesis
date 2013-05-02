#ifndef GPUDIXXIMULT1D_H
#define GPUDIXXIMULT1D_H

#include "../../../common/utils.h"
#include "../../../common/GPUAlgorithm.h"
#include "../../MatrixAlgorithm.h"

namespace gpu
{
    namespace dixxi
    {
        template<typename T>
        class Mult1D : public GPUAlgorithm<T>, public MatrixAlgorithm
        {
            public:
                const string getName() override
                {
                    return "Matrix multiplication 1D";
                }

                void init(Context* context) override
                {
                    Program* program = context->createProgram("gpu/dixxi/Mult1D.cl", "-D T=" + getTypeName<T>());
                    kernel = program->createKernel("Mult");
                    delete program;
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
                {
                    size_t elementCount = size * size;

                    a = context->createBuffer(CL_MEM_READ_ONLY, elementCount * sizeof(T));
                    queue->enqueueWrite(a, data, 0, elementCount * sizeof(T));

                    b = context->createBuffer(CL_MEM_READ_ONLY, elementCount * sizeof(T));
                    queue->enqueueWrite(b, data + elementCount, 0, elementCount * sizeof(T));

                    c = context->createBuffer(CL_MEM_WRITE_ONLY, elementCount * sizeof(T));
                }

                void run(CommandQueue* queue, size_t workGroupSize, size_t size) override
                {
                    kernel->setArg(0, a);
                    kernel->setArg(1, b);
                    kernel->setArg(2, c);
                    kernel->setArg(3, (cl_uint)size);

                    size_t globalWorkSizes[] = { roundToMultiple(size * size, workGroupSize) };
                    size_t localWorkSizes[] = { workGroupSize };

                    queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
                }

                void download(CommandQueue* queue, T* result, size_t size) override
                {
                    queue->enqueueRead(c, result, 0, size * size * sizeof(T));
                }

                void cleanup() override
                {
                    delete kernel;
                    delete a;
                    delete b;
                    delete c;
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

#endif // GPUDIXXIMULT1D_H
