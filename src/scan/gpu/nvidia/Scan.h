#pragma once

#include "../../ScanAlgorithm.h"
#include "../../../common/CLAlgorithm.h"
#include "../../../common/utils.h"

namespace gpu
{
    namespace nvidia
    {
        /**
         * From: http://developer.download.nvidia.com/compute/cuda/4_2/rel/sdk/website/OpenCL/html/samples.html
         */
        template<typename T>
        class Scan : public CLAlgorithm<T>, public ScanAlgorithm
        {
            public:
                static const size_t WORKGROUP_SIZE = 256;
                static const size_t MIN_LARGE_ARRAY_SIZE = 8 * WORKGROUP_SIZE;
                static const size_t MAX_LARGE_ARRAY_SIZE = 4 * WORKGROUP_SIZE * WORKGROUP_SIZE;

                static const size_t MAX_BATCH_ELEMENTS = 64 * 1048576;


                const string getName() override
                {
                    return "Scan NVIDIA (exclusiv)";
                }

                bool isInclusiv() override
                {
                    return false;
                }

                void init() override
                {
                    stringstream ss;
                    ss << "-D T=" << getTypeName<T>() << " -D T4=" << getTypeName<T>() << "4" << " -D WORKGROUP_SIZE=" << WORKGROUP_SIZE;

                    Program* program = context->createProgram("gpu/nvidia/Scan.cl", ss.str());
                    exclusiveLocalKernel1 = program->createKernel("scanExclusiveLocal1");
                    exclusiveLocalKernel2 = program->createKernel("scanExclusiveLocal2");
                    uniformUpdateKernel = program->createKernel("uniformUpdate");
                    delete program;

                    buffer = context->createBuffer(CL_MEM_READ_WRITE, (MAX_BATCH_ELEMENTS / (4 * WORKGROUP_SIZE)) * sizeof(T));
                }

                void upload(size_t workGroupSize, T* data, size_t size) override
                {
                    bufferSize = roundToPowerOfTwo(size);

                    source = context->createBuffer(CL_MEM_READ_WRITE, bufferSize * sizeof(T));
                    queue->enqueueWrite(source, data);

                    destination = context->createBuffer(CL_MEM_READ_WRITE, bufferSize * sizeof(T));
                }

                void scanExclusiveLocal1(CommandQueue* queue, size_t n, size_t size)
                {
                    exclusiveLocalKernel1->setArg(0, destination);
                    exclusiveLocalKernel1->setArg(1, source);
                    exclusiveLocalKernel1->setArg(2, 2 * WORKGROUP_SIZE * sizeof(T), nullptr);
                    exclusiveLocalKernel1->setArg(3, (cl_uint)size);

                    size_t localWorkSizes[] = { WORKGROUP_SIZE };
                    size_t globalWorkSizes[] = { (n * size) / 4 };

                    queue->enqueueKernel(exclusiveLocalKernel1, 1, globalWorkSizes, localWorkSizes);
                }

                void scanExclusiveLocal2(CommandQueue* queue, size_t n, size_t size)
                {
                    size_t elements = n * size;
                    exclusiveLocalKernel2->setArg(0, buffer);
                    exclusiveLocalKernel2->setArg(1, destination);
                    exclusiveLocalKernel2->setArg(2, source);
                    exclusiveLocalKernel2->setArg(3, 2 * WORKGROUP_SIZE * sizeof(T), nullptr);
                    exclusiveLocalKernel2->setArg(4, (cl_uint)elements);
                    exclusiveLocalKernel2->setArg(5, (cl_uint)size);

                    size_t localWorkSizes[] = { WORKGROUP_SIZE };
                    size_t globalWorkSizes[] = { roundToMultiple(elements, WORKGROUP_SIZE) };

                    queue->enqueueKernel(exclusiveLocalKernel2, 1,globalWorkSizes, localWorkSizes);
                }

                void uniformUpdate(CommandQueue* queue, size_t n)
                {
                    uniformUpdateKernel->setArg(0, destination);
                    uniformUpdateKernel->setArg(1, buffer);

                    size_t localWorkSizes[] = { WORKGROUP_SIZE };
                    size_t globalWorkSizes[] = { n * WORKGROUP_SIZE };

                    queue->enqueueKernel(uniformUpdateKernel, 1, globalWorkSizes, localWorkSizes);
                }

                void run(size_t workGroupSize, size_t size) override
                {
                    size_t batchSize = 1;
                    size_t arrayLength = bufferSize;

                    //Check supported size range
                    if((arrayLength < MIN_LARGE_ARRAY_SIZE) || (arrayLength > MAX_LARGE_ARRAY_SIZE))
                        throw OpenCLException("Element count is out of supported size");

                    //Check total batch size limit
                    if(batchSize * arrayLength > MAX_BATCH_ELEMENTS)
                        throw OpenCLException("Batch element count is out of supported size");

                    scanExclusiveLocal1(queue, (batchSize * arrayLength) / (4 * WORKGROUP_SIZE), 4 * WORKGROUP_SIZE);

                    scanExclusiveLocal2(queue, batchSize, arrayLength / (4 * WORKGROUP_SIZE));

                    uniformUpdate(queue,(batchSize * arrayLength) / (4 * WORKGROUP_SIZE));
                }

                void download(T* result, size_t size) override
                {
                    queue->enqueueRead(destination, result, 0, size * sizeof(T));

                    delete source;
                    delete destination;
                }

                void cleanup() override
                {
                    delete exclusiveLocalKernel1;
                    delete exclusiveLocalKernel2;
                    delete uniformUpdateKernel;
                    delete buffer;
                }

                virtual ~Scan() {}

            private:
                size_t bufferSize;
                Kernel* exclusiveLocalKernel1;
                Kernel* exclusiveLocalKernel2;
                Kernel* uniformUpdateKernel;
                Buffer* source;
                Buffer* destination;
                Buffer* buffer;
        };
    }
}
