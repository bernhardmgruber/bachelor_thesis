#ifndef GPUDIXXITRANSFORM_H
#define GPUDIXXITRANSFORM_H

#include "../../../common/GPUAlgorithm.h"
#include "../../MeshTransformAlgorithm.h"

namespace gpu
{
    namespace dixxi
    {
        template <typename T>
        class Transform : public GPUAlgorithm<T>, public MeshTransformAlgorithm
        {
            public:
                const string getName() override
                {
                    return "Transform";
                }

                void init(Context* context) override
                {
                    stringstream ss;
                    ss << "-D T=" << getTypeName<T>() << " -D MATRIX_SIZE=" << MATRIX_SIZE;
                    Program* program = context->createProgram("gpu/dixxi/Transform.cl", ss.str());
                    kernel = program->createKernel("Transform");
                    delete program;
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
                {
                    matrixBuffer = context->createBuffer(CL_MEM_READ_ONLY, MATRIX_SIZE * sizeof(T));
                    queue->enqueueWrite(matrixBuffer, data);

                    vertexBuffer = context->createBuffer(CL_MEM_READ_WRITE, size * 3 * sizeof(T));
                    queue->enqueueWrite(vertexBuffer, data + MATRIX_SIZE);
                }
                void run(CommandQueue* queue, size_t workGroupSize, size_t size) override
                {
                    kernel->setArg(0, matrixBuffer);
                    kernel->setArg(1, vertexBuffer);

                    size_t globalWorkSizes[] = { size };
                    size_t localWorkSizes[] = { workGroupSize };

                    queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
                }

                void download(CommandQueue* queue, T* result, size_t size) override
                {
                    queue->enqueueWrite(vertexBuffer, result);
                    delete matrixBuffer;
                    delete vertexBuffer;
                }

                void cleanup() override
                {
                    delete kernel;
                }

                virtual ~Transform() {}

            private:
                Kernel* kernel;
                Buffer* matrixBuffer;
                Buffer* vertexBuffer;
        };
    }
}

#endif // CPUDIXXITRANSFORM_H
