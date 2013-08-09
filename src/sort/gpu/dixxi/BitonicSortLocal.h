#pragma once

#include "../../../common/CLAlgorithm.h"
#include "../../SortAlgorithm.h"

using namespace std;

namespace gpu
{
    namespace dixxi
    {
        /**
        * From: http://www.bealto.com/gpu-sorting_intro.html
        * ParallelBitonicSortLocalOptim
        */
        template<typename T>
        class BitonicSortLocal : public CLAlgorithm<T>, public SortAlgorithm
        {
        public:
            const string getName() override
            {
                return "Bitonic sort local (dixxi Bealto)";
            }

            bool isInPlace() override
            {
                return false;
            }

            void init() override
            {
                Program* program = context->createProgram("gpu/dixxi/BitonicSortLocal.cl", "-I gpu/dixxi/ -D T=" + getTypeName<T>());
                kernelLocal = program->createKernel("BitonicSortLocal");
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
                size_t nThreads = bufferSize / 2;
                workGroupSize = min(workGroupSize, nThreads);

                size_t globalWorkSizes[1] = { nThreads };
                size_t localWorkSizes[1] = { workGroupSize };

                for (cl_int startInc = 1; startInc < bufferSize; startInc <<= 1)
                {
                    for (cl_int inc = startInc; inc > 0; inc >>= 1)
                    {
                        if(inc <= workGroupSize)
                        {
                            // finish this box with the local kernel
                            kernelLocal->setArg(0, buffer);
                            kernelLocal->setArg(1, sizeof(T) * workGroupSize * 2, nullptr);
                            kernelLocal->setArg(2, inc);
                            kernelLocal->setArg(3, startInc << 1);

                            queue->enqueueKernel(kernelLocal, 1, globalWorkSizes, localWorkSizes);

                            break;
                        }

                        kernel->setArg(0, buffer);
                        kernel->setArg(1, inc);          
                        kernel->setArg(2, startInc << 1);  

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

            virtual ~BitonicSortLocal() {}

        private:
            Kernel* kernel;
            Kernel* kernelLocal;
            Buffer* buffer;
            size_t bufferSize;
        };
    }
}
