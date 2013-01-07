#ifndef GPUNVIDIABITONICSORT_H
#define GPUNVIDIABITONICSORT_H

#include "../../../common/GPUAlgorithm.h"
#include "../../SortAlgorithm.h"

#include <sstream>

using namespace std;

namespace gpu
{
    namespace nvidia
    {
        template<typename T>
        class BitonicSort : public GPUAlgorithm<T>, public SortAlgorithm
        {
            public:
                const static size_t LOCAL_SIZE_LIMIT = 512;

                const string getName() override
                {
                    return "Bitonicsort (NVIDIA)";
                }

                bool isInPlace() override
                {
                    return false;
                }

                void init(Context* context) override
                {
                    stringstream ss;
                    ss << "-D T=" << getTypeName<T>() << " -D LOCAL_SIZE_LIMIT=" << LOCAL_SIZE_LIMIT;

                    Program* program = context->createProgram("gpu/nvidia/BitonicSort.cl", ss.str());
                    ckBitonicSortLocal   = program->createKernel("bitonicSortLocal");
                    ckBitonicSortLocal1  = program->createKernel("bitonicSortLocal1");
                    ckBitonicMergeGlobal = program->createKernel("bitonicMergeGlobal");
                    ckBitonicMergeLocal  = program->createKernel("bitonicMergeLocal");
                    delete program;

                    size_t szBitonicSortLocal = ckBitonicSortLocal->getWorkGroupSize();
                    size_t szBitonicSortLocal1 = ckBitonicSortLocal1->getWorkGroupSize();
                    size_t szBitonicMergeLocal = ckBitonicMergeLocal->getWorkGroupSize();

                    if((szBitonicSortLocal < (LOCAL_SIZE_LIMIT / 2)) || (szBitonicSortLocal1 < (LOCAL_SIZE_LIMIT / 2)) || (szBitonicMergeLocal < (LOCAL_SIZE_LIMIT / 2)))
                        throw OpenCLException("Minimum required work group size: " + (LOCAL_SIZE_LIMIT / 2));
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
                {
                    bufferSize = pow2roundup(size);

                    d_InputKey  = context->createBuffer(CL_MEM_READ_ONLY, bufferSize * sizeof(T));
                    //d_InputVal  = context->createBuffer(CL_MEM_READ_ONLY, bufferSize * sizeof(T));
                    d_OutputKey = context->createBuffer(CL_MEM_READ_WRITE, bufferSize * sizeof(T));
                    //d_OutputVal = context->createBuffer(CL_MEM_READ_WRITE, bufferSize * sizeof(T));

                    queue->enqueueWrite(d_InputKey, data, 0, size * sizeof(T));
                }

                void run(CommandQueue* queue, size_t workGroupSize, size_t size) override
                {
                    size_t batch = 1;
                    size_t arrayLength = bufferSize;
                    unsigned int dir = 1;

                    //Launch bitonicSortLocal1
                    ckBitonicSortLocal1->setArg(0, d_OutputKey);
                    //ckBitonicSortLocal1->setArg(1, d_OutputVal);
                    ckBitonicSortLocal1->setArg(1, d_InputKey);
                    //ckBitonicSortLocal1->setArg(3, d_InputVal);

                    size_t localWorkSize = LOCAL_SIZE_LIMIT / 2;
                    size_t globalWorkSize = batch * arrayLength / 2;
                    queue->enqueueKernel(ckBitonicSortLocal1, 1, &globalWorkSize, &localWorkSize);

                    for(size_t size = 2 * LOCAL_SIZE_LIMIT; size <= arrayLength; size <<= 1)
                    {
                        for(unsigned stride = size / 2; stride > 0; stride >>= 1)
                        {
                            if(stride >= LOCAL_SIZE_LIMIT)
                            {
                                //Launch bitonicMergeGlobal
                                ckBitonicMergeGlobal->setArg(0, d_OutputKey);
                                //ckBitonicMergeGlobal->setArg(1, d_OutputVal);
                                ckBitonicMergeGlobal->setArg(1, d_OutputKey);
                                //ckBitonicMergeGlobal->setArg(3, d_OutputVal);
                                ckBitonicMergeGlobal->setArg(2, (cl_uint)arrayLength);
                                ckBitonicMergeGlobal->setArg(3, (cl_uint)size);
                                ckBitonicMergeGlobal->setArg(4, (cl_uint)stride);
                                ckBitonicMergeGlobal->setArg(5, (cl_uint)dir);

                                globalWorkSize = batch * arrayLength / 2;
                                queue->enqueueKernel(ckBitonicMergeGlobal, 1, &globalWorkSize, nullptr); // no local worksize
                            }
                            else
                            {
                                //Launch bitonicMergeLocal
                                ckBitonicMergeLocal->setArg(0, d_OutputKey);
                                //ckBitonicMergeLocal->setArg(1, d_OutputVal);
                                ckBitonicMergeLocal->setArg(1, d_OutputKey);
                                //ckBitonicMergeLocal->setArg(3, d_OutputVal);
                                ckBitonicMergeLocal->setArg(2, (cl_uint)arrayLength);
                                ckBitonicMergeLocal->setArg(3, (cl_uint)stride);
                                ckBitonicMergeLocal->setArg(4, (cl_uint)size);
                                ckBitonicMergeLocal->setArg(5, (cl_uint)dir);

                                localWorkSize  = LOCAL_SIZE_LIMIT / 2;
                                globalWorkSize = batch * arrayLength / 2;

                                queue->enqueueKernel(ckBitonicMergeLocal, 1, &globalWorkSize, &localWorkSize);
                                break;
                            }
                        }
                    }
                }

                void download(CommandQueue* queue, T* result, size_t size) override
                {
                    queue->enqueueRead(d_OutputKey, result, 0, size * sizeof(T));

                    delete d_InputKey;
                    //delete d_InputVal;
                    delete d_OutputKey;
                    //delete d_OutputVal;
                }

                void cleanup() override
                {
                    delete ckBitonicSortLocal;
                    delete ckBitonicSortLocal1;
                    delete ckBitonicMergeGlobal;
                    delete ckBitonicMergeLocal;
                }

                virtual ~BitonicSort() {}

            private:
                Kernel* ckBitonicSortLocal;
                Kernel* ckBitonicSortLocal1;
                Kernel* ckBitonicMergeGlobal;
                Kernel* ckBitonicMergeLocal;

                Buffer* d_InputKey;
                //Buffer* d_InputVal;
                Buffer* d_OutputKey;
                //Buffer* d_OutputVal;

                size_t bufferSize;
        };
    }
}

#endif // GPUNVIDIABITONICSORT_H
