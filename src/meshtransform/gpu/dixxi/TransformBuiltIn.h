#ifndef GPUDIXXITRANSFORMBUILTIN_H
#define GPUDIXXITRANSFORMBUILTIN_H

#include "../../../common/GPUAlgorithm.h"
#include "../../MeshTransformAlgorithm.h"
#include "../../../common/utils.h"

namespace gpu
{
    namespace dixxi
    {
        template <typename T>
        class TransformBuiltIn : public GPUAlgorithm<T>, public MeshTransformAlgorithm
        {
            public:
                static const size_t BLOCK_SIZE = 4;

                const string getName() override
                {
                    return "Transform built in";
                }

                void init(Context* context) override
                {
                    stringstream ss;
                    ss << "-D T=" << getTypeName<T>() << " -D BLOCK_SIZE=" << BLOCK_SIZE;
                    Program* program = context->createProgram("gpu/dixxi/TransformBuiltIn.cl", ss.str());
                    kernel = program->createKernel("TransformBuiltIn");
                    delete program;
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
                {
                    adaptedSize = roundToMultiple(size, workGroupSize * BLOCK_SIZE);

                    matrixBuffer = context->createBuffer(CL_MEM_READ_ONLY, MATRIX_SIZE * sizeof(T));
                    queue->enqueueWrite(matrixBuffer, data);

                    vertexBuffer = context->createBuffer(CL_MEM_READ_WRITE, adaptedSize * 3 * sizeof(T));
                    queue->enqueueWrite(vertexBuffer, data + MATRIX_SIZE, 0, size * 3 * sizeof(T));
                }

                void run(CommandQueue* queue, size_t workGroupSize, size_t size) override
                {
                    kernel->setArg(0, matrixBuffer);
                    kernel->setArg(1, vertexBuffer);

                    size_t globalWorkSizes[] = { adaptedSize };
                    size_t localWorkSizes[] = { workGroupSize };

                    queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
                }

                void download(CommandQueue* queue, T* result, size_t size) override
                {
                    queue->enqueueRead(vertexBuffer, result, 0, size * 3 * sizeof(T));
                    delete matrixBuffer;
                    delete vertexBuffer;
                }

                void cleanup() override
                {
                    delete kernel;
                }

                virtual ~TransformBuiltIn() {}

            private:
                Kernel* kernel;
                Buffer* matrixBuffer;
                Buffer* vertexBuffer;
                size_t adaptedSize;
        };
    }
}

#endif // GPUDIXXITRANSFORMBUILTIN_H
