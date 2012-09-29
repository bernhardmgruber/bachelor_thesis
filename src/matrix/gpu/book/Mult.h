#ifndef GPUBOOKMULT_H
#define GPUBOOKMULT_H

#include "../../../common/utils.h"
#include "../../../common/GPUAlgorithm.h"
#include "../../MatrixAlgorithm.h"

namespace gpu
{
    namespace book
    {
        template<typename T>
        class Mult : public GPUAlgorithm<T>, public MatrixAlgorithm
        {
            public:
                const string getName() override
                {
                    return "Matrix multiplication (Book)";
                }

                void init(Context* context) override
                {
                    Program* program = context->createProgram("gpu/book/Mult.cl", "-D T=" + getTypeName<T>());
                    kernel = program->createKernel("Mult");
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
                {
                    size_t elementCount = size * size;
                    adjustedSize = roundToMultiple(elementCount, workGroupSize);

                    a = context->createBuffer(CL_MEM_READ_ONLY, adjustedSize * sizeof(T));
                    queue->enqueueWrite(a, data, 0, elementCount * sizeof(T));

                    b = context->createBuffer(CL_MEM_READ_ONLY, adjustedSize * sizeof(T));
                    queue->enqueueWrite(b, (T*)data + elementCount, 0, elementCount * sizeof(T));

                    c = context->createBuffer(CL_MEM_WRITE_ONLY, adjustedSize * sizeof(T));
                }

                void run(CommandQueue* queue, size_t workGroupSize, size_t size) override
                {
                    kernel->setArg(0, a);
                    kernel->setArg(1, b);
                    kernel->setArg(2, c);
                    kernel->setArg(3, (cl_uint)size);

                    size_t globalWorkSizes[] = { adjustedSize };
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

                virtual ~Mult() {}

            private:
                Kernel* kernel;
                Buffer* a;
                Buffer* b;
                Buffer* c;
                size_t adjustedSize;
        };
    }
}


#endif
