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

                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
                {

                }
                void run(CommandQueue* queue, size_t workGroupSize, size_t size) override
                {

                }
                void download(CommandQueue* queue, T* result, size_t size) override
                {

                }
                void cleanup() override
                {

                }

                virtual ~Transform() {}
        };
    }
}

#endif // CPUDIXXITRANSFORM_H
