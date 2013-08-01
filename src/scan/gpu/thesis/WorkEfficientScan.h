#pragma once

#include "../../ScanAlgorithm.h"
#include "../../../common/CLAlgorithm.h"

#include "../../../common/utils.h"

namespace gpu
{
    namespace thesis
    {
        /**
        * Full array version of: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
        * Chapter: 39.2.2 A Work-Efficient Parallel Scan
        */
        template<typename T>
        class WorkEfficientScan : public CLAlgorithm<T>, public ScanAlgorithm
        {
            static_assert(is_same<T, cl_int>::value, "Thesis algorithms only support int");

        public:
            const string getName() override
            {
                return "Work Efficient Scan (THESIS dixxi GPU Gems) (exclusiv)";
            }

            bool isInclusiv() override
            {
                return false;
            }

            void init() override
            {
                Program* program = context->createProgram("gpu/thesis/WorkEfficientScan.cl");
                upSweepKernel = program->createKernel("UpSweep");
                setLastZeroKernel = program->createKernel("SetLastZero");
                downSweepKernel = program->createKernel("DownSweep");
                delete program;
            }

            void upload(size_t workGroupSize, T* data, size_t size) override
            {
                bufferSize = roundToPowerOfTwo(size);

                buffer = context->createBuffer(CL_MEM_READ_WRITE, bufferSize * sizeof(T));
                queue->enqueueWrite(buffer, data);
            }

            void run(size_t workGroupSize, size_t size) override
            {
                size_t globalWorkSizes[] = { bufferSize };
                size_t localWorkSizes[] = { min(workGroupSize, bufferSize) };

                // upsweep (reduce)
                for(cl_uint offset = 1; offset < bufferSize; offset <<= 1)
                {
                    cl_uint stride = 2 * offset;

                    upSweepKernel->setArg(0, buffer);
                    upSweepKernel->setArg(1, offset);
                    upSweepKernel->setArg(2, stride);

                    queue->enqueueKernel(upSweepKernel, 1, globalWorkSizes, localWorkSizes);
                }

                // set last element to zero
                setLastZeroKernel->setArg(0, buffer);
                setLastZeroKernel->setArg(1, (cl_uint)(bufferSize - 1));
                queue->enqueueTask(setLastZeroKernel);

                // downsweep
                for(cl_uint offset = (cl_uint)bufferSize >> 1; offset >= 1; offset >>= 1)
                {
                    cl_uint stride = 2 * offset;

                    downSweepKernel->setArg(0, buffer);
                    downSweepKernel->setArg(1, offset);
                    downSweepKernel->setArg(2, stride);

                    queue->enqueueKernel(downSweepKernel, 1, globalWorkSizes, localWorkSizes);
                }
            }

            void download(T* result, size_t size) override
            {
                queue->enqueueRead(buffer, result, 0, size * sizeof(T));
                delete buffer;
            }

            void cleanup() override
            {
                delete upSweepKernel;
                delete downSweepKernel;
                delete setLastZeroKernel;
            }

            virtual ~WorkEfficientScan() {}

        private:
            size_t bufferSize;
            Kernel* upSweepKernel;
            Kernel* downSweepKernel;
            Kernel* setLastZeroKernel;
            Buffer* buffer;
        };
    }
}
