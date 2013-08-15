#pragma once

#include <algorithm>

#include "../../../common/CLAlgorithm.h"
#include "../../SortAlgorithm.h"

using namespace std;

namespace gpu
{
    namespace thesis
    {
        /**
        * From: http://www.bealto.com/gpu-sorting_intro.html
        * ParallelBitonicSortB
        */
        template<typename T>
        class BitonicSort : public CLAlgorithm<T>, public SortAlgorithm
        {
            static_assert(is_same<T, cl_uint>::value, "Thesis algorithms only support 32 bit unsigned int");

        public:
            const string getName() override
            {
                return "Bitonic sort (THESIS dixxi Bealto)";
            }

            bool isInPlace() override
            {
                return false;
            }

            void init() override
            {
                Program* program = context->createProgram("gpu/thesis/BitonicSort.cl", "-D T=" + getTypeName<T>());
                kernel = program->createKernel("BitonicSort");
                delete program;
            }

            void upload(size_t workGroupSize, T* data, size_t size) override
            {
                bufferSize = roundToPowerOfTwo(size);

                buffer = context->createBuffer(CL_MEM_READ_WRITE, bufferSize * sizeof(T));

                if(bufferSize != size)
                {
                    queue->enqueueWrite(buffer, data, 0, size * sizeof(T));
                    queue->enqueueFill(buffer, numeric_limits<T>::max(), size * sizeof(T), (bufferSize - size) * sizeof(T));
                }
                else
                    queue->enqueueWrite(buffer, data);
            }

            void run(size_t workGroupSize, size_t size) override
            {
                for (cl_uint startInc = 1; startInc < bufferSize; startInc <<= 1)
                {
                    for (cl_uint inc = startInc; inc > 0; inc >>= 1)
                    {
                        kernel->setArg(0, buffer);
                        kernel->setArg(1, inc);          
                        kernel->setArg(2, startInc << 1);  

                        size_t nThreads = bufferSize / 2;
                        size_t globalWorkSizes[1] = { nThreads };
                        size_t localWorkSizes[1] = { min(workGroupSize, nThreads) };

                        queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);
                    }
                }
            }

            void download(T* result, size_t size) override
            {
                queue->enqueueRead(buffer, result, 0, size * sizeof(T));
                delete buffer;
            }

            void cleanup() override
            {
                delete kernel;
            }

            virtual ~BitonicSort() {}

        private:
            Kernel* kernel;
            Buffer* buffer;
            size_t bufferSize;
        };
    }
}
