#pragma once

#include <algorithm>

#include "../../../common/CLAlgorithm.h"
#include "../../SortAlgorithm.h"

using namespace std;

namespace gpu
{
    namespace dixxi
    {
        /**
        * From: http://www.bealto.com/gpu-sorting_intro.html
        * ParallelBitonicSortB16
        */
        template<typename T>
        class BitonicSortFusion : public CLAlgorithm<T>, public SortAlgorithm
        {
        public:
            const string getName() override
            {
                return "Bitonic sort fusion (dixxi Bealto)";
            }

            bool isInPlace() override
            {
                return false;
            }

            void init() override
            {
                Program* program = context->createProgram("gpu/dixxi/BitonicSortFusion.cl", "-D T=" + getTypeName<T>());
                kernel2 = program->createKernel("BitonicSortFusion2");
                kernel4 = program->createKernel("BitonicSortFusion4");
                kernel8 = program->createKernel("BitonicSortFusion8");
                kernel16 = program->createKernel("BitonicSortFusion16");
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
                    for (cl_uint inc = startInc; inc > 0; )
                    {
                        int ninc = 0;
                        Kernel* kernel;

                        if (inc >= 8 && ninc == 0)
                        {
                            kernel = kernel16;
                            ninc = 4;
                        }

                        if (inc >= 4 && ninc == 0)
                        {
                            kernel = kernel8;
                            ninc = 3;
                        }

                        if (inc >= 2 && ninc == 0)
                        {
                            kernel = kernel4;
                            ninc = 2;
                        }

                        if (ninc == 0)
                        {
                            kernel = kernel2;
                            ninc = 1;
                        }

                        kernel->setArg(0, buffer);
                        kernel->setArg(1, inc); 
                        kernel->setArg(2, startInc << 1); 

                        size_t nThreads = bufferSize >> ninc;
                        size_t globalWorkSizes[1] = { nThreads };
                        size_t localWorkSizes[1] = { min(workGroupSize, nThreads) };

                        queue->enqueueKernel(kernel, 1, globalWorkSizes, localWorkSizes);

                        inc >>= ninc;
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
                delete kernel2;
                delete kernel4;
                delete kernel8;
                delete kernel16;
            }

            virtual ~BitonicSortFusion() {}

        private:
            Kernel* kernel2;
            Kernel* kernel4;
            Kernel* kernel8;
            Kernel* kernel16;
            Buffer* buffer;
            size_t bufferSize;
        };
    }
}
