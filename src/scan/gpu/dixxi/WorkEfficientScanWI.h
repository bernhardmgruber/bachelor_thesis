#pragma once

#include "../../ScanAlgorithm.h"
#include "../../../common/CLAlgorithm.h"

#include "../../../common/utils.h"

namespace gpu
{
    namespace dixxi
    {
        /**
        * Full array version of: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
        * Chapter: 39.2.2 A Work-Efficient Parallel Scan
        */
        template<typename T>
        class WorkEfficientScanWI : public CLAlgorithm<T>, public ScanAlgorithm
        {
        public:
            const string getName() override
            {
                return "Work Efficient Scan WI (dixxi GPU Gems) (exclusiv)";
            }

            bool isInclusiv() override
            {
                return false;
            }

            void init() override
            {
                Program* program = context->createProgram("gpu/dixxi/WorkEfficientScanWI.cl", "-D T=" + getTypeName<T>());
                upSweepKernel = program->createKernel("UpSweep");
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
                // upsweep (reduce)
                for(size_t offset = 1, nodes = bufferSize >> 1; offset < bufferSize; offset <<= 1, nodes >>= 1)
                {
                    upSweepKernel->setArg(0, buffer);
                    upSweepKernel->setArg(1, (cl_uint)offset);

                    size_t globalWorkSizes[] = { nodes };
                    size_t localWorkSizes[] = { min(workGroupSize, nodes) };

                    queue->enqueueKernel(upSweepKernel, 1, globalWorkSizes, localWorkSizes);
                }

                // set last element to zero
                cl_uint zero = 0;
                queue->enqueueWrite(buffer, &zero, (bufferSize - 1) * sizeof(cl_uint), sizeof(cl_uint));

                // downsweep
                for(size_t offset = bufferSize >> 1, nodes = 1; offset >= 1; offset >>= 1, nodes <<= 1)
                {
                    downSweepKernel->setArg(0, buffer);
                    downSweepKernel->setArg(1, (cl_uint)offset);

                    size_t globalWorkSizes[] = { nodes };
                    size_t localWorkSizes[] = { min(workGroupSize, nodes) };

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
            }

            virtual ~WorkEfficientScanWI() {}

        private:
            size_t bufferSize;
            Kernel* upSweepKernel;
            Kernel* downSweepKernel;
            Buffer* buffer;
        };
    }
}
