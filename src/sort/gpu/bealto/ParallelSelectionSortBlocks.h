#pragma once

#include <sstream>

#include "../../../common/CLAlgorithm.h"
#include "../../SortAlgorithm.h"

using namespace std;

namespace gpu
{
    namespace bealto
    {
        /**
         * From: http://www.bealto.com/gpu-sorting_intro.html
         */
        template<typename T>
        class ParallelSelectionSortBlocks : public CLAlgorithm<T>, public SortAlgorithm
        {
            public:
                ParallelSelectionSortBlocks()
                {
                    blockFactor = 1;
                }

                virtual ~ParallelSelectionSortBlocks()
                {
                }

                const string getName() override
                {
                    return "Parallel selection sort blocks (Bealto)";
                }

                bool isInPlace() override
                {
                    return false;
                }

                void init() override
                {
                    stringstream options;
                    options << "-D BLOCK_FACTOR=" << blockFactor << " -D T=" << getTypeName<T>();
                    Program* program = context->createProgram("gpu/bealto/ParallelSelectionSortBlocks.cl", options.str());
                    kernel = program->createKernel("ParallelSelectionSortBlocks");
                    delete program;
                }

                void upload(size_t workGroupSize, T* data, size_t size) override
                {
                    in = context->createBuffer(CL_MEM_READ_ONLY, sizeof(T) * size);
                    queue->enqueueWrite(in, data);
                    out = context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * size);
                }

                void run(size_t workGroupSize, size_t size) override
                {
                    kernel->setArg(0, in);
                    kernel->setArg(1, out);
                    kernel->setArg(2, sizeof(cl_uint) * workGroupSize * blockFactor, nullptr);
                    size_t globalWorkSizes[1] = { size };
                    size_t localWorkSizes[1] = { workGroupSize };
                    queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
                }

                void download(T* result, size_t size) override
                {
                    queue->enqueueRead(out, result);
                    delete in;
                    delete out;
                }

                void cleanup() override
                {
                    delete kernel;
                }

            private:
                size_t blockFactor;
                Kernel* kernel;
                Buffer* in;
                Buffer* out;
        };
    }
}
