#ifndef GPUDIXXIMULTIMAGE_H
#define GPUDIXXIMULTIMAGE_H

#include "../../../common/utils.h"
#include "../../../common/GPUAlgorithm.h"
#include "../../MatrixAlgorithm.h"

namespace gpu
{
    namespace dixxi
    {
        template<typename T>
        class MultImage : public GPUAlgorithm<T>, public MatrixAlgorithm
        {
            public:
                const string getName() override
                {
                    return "Matrix multiplication (texture unit)";
                }

                void init(Context* context) override
                {
                    if(context->getInfo<cl_bool>(CL_DEVICE_IMAGE_SUPPORT) == false)
                        throw OpenCLException("Images are not supported!");

                    Program* program = context->createProgram("gpu/dixxi/MultImage.cl", "-D T=" + getTypeName<T>());
                    kernel = program->createKernel("Mult");
                    delete program;
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
                {
                    cl_image_desc descriptor;
                    memset(&descriptor, 0, sizeof(cl_image_desc));
                    descriptor.image_type = CL_MEM_OBJECT_IMAGE2D;
                    descriptor.image_width = size;
                    descriptor.image_height = size;
                    descriptor.image_depth = 1;

                    cl_image_format format;
                    format.image_channel_order = CL_R;
                    format.image_channel_data_type = CL_FLOAT;

                    a = context->createImage(CL_MEM_READ_ONLY, format, descriptor);
                    b = context->createImage(CL_MEM_READ_ONLY, format, descriptor);
                    c = context->createImage(CL_MEM_WRITE_ONLY, format, descriptor);

                    size_t elementCount = size * size;

                    float* mappedA = (float*)queue->enqueueMap(a, CL_MAP_WRITE);
                    memcpy(mappedA, data, elementCount * sizeof(float));
                    queue->enqueueUnmap(a, mappedA);

                    float* mappedB = (float*)queue->enqueueMap(b, CL_MAP_WRITE);
                    memcpy(mappedB, data + elementCount, elementCount * sizeof(float));
                    queue->enqueueUnmap(b, mappedB);
                }

                void run(CommandQueue* queue, size_t workGroupSize, size_t size) override
                {
                    workGroupSize = 16;

                    kernel->setArg(0, a);
                    kernel->setArg(1, b);
                    kernel->setArg(2, c);
                    kernel->setArg(3, (cl_uint)size);

                    size_t adjusted = roundToMultiple(size, workGroupSize);
                    size_t globalWorkSizes[] = { adjusted, adjusted };
                    size_t localWorkSizes[] = { workGroupSize, workGroupSize };

                    queue->enqueueKernel(kernel, 2, globalWorkSizes, localWorkSizes);
                }

                void download(CommandQueue* queue, T* result, size_t size) override
                {
                    queue->enqueueRead(c, result);
                }

                void cleanup() override
                {
                    delete kernel;
                    delete a;
                    delete b;
                    delete c;
                }

                virtual ~MultImage() {}

            private:
                Kernel* kernel;
                Image* a;
                Image* b;
                Image* c;
        };
    }
}

#endif // GPUDIXXIMULTIMAGE_H
