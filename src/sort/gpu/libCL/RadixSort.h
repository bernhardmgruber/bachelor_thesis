// Copyright [2011] [Geist Software Labs Inc.]
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LibCLRadixSort
#define LibCLRadixSort

#include "../../GPUSortingAlgorithm.h"
//#include "../../OpenCL.h"

using namespace std;

#define CBITS 4

namespace gpu
{
    namespace libcl
    {
        /**
         * From: http://www.libcl.org/
         * Modified by Bernhard Manfred Gruber to be key only.
         */
        template<typename T, size_t count>
        class RadixSort : public GPUSortingAlgorithm<T, count>
        {
            public:
                string getName() override
                {
                    return "Radix sort (LibCL)";
                }

                void init(Context* context) override
                {
                    program = context->createProgram("gpu/libCL/RadixSort.cl");

                    clBlockSort = program->createKernel("clBlockSort");
                    clBlockScan = program->createKernel("clBlockScan");
                    clBlockPrefix = program->createKernel("clBlockPrefix");
                    clReorder = program->createKernel("clReorder");
                }

                void upload(Context* context, T* data) override
                {
                    size_t cBits = CBITS;
                    size_t cBlockSize = context->getInfoSize(CL_DEVICE_MAX_WORK_GROUP_SIZE);

                    size_t lBlockCount = ceil((float)count / cBlockSize);

                    bfKey = context->createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(T) * count, data);
                    bfTempKey = context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * count);
                    //bfTempVal = context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * count);
                    bfBlockScan = context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * lBlockCount * (1 << cBits));
                    bfBlockOffset = context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * lBlockCount * (1 << cBits));
                    bfBlockSum = context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * cBlockSize);
                }

                void sort(CommandQueue* queue, size_t workGroupSize) override
                {
                    int iStartBit = 0;
                    int iEndBit = 32;

                    size_t cBits = CBITS;
                    size_t cBlockSize = workGroupSize;

                    if ((iEndBit - iStartBit) % cBits != 0)
                    {
                        cerr << "end bit(" << iEndBit << ") - start bit(" << iStartBit << ") must be divisible by 4";
                        return;
                    }

                    size_t lBlockCount = ceil((float)count / cBlockSize);

                    size_t lGlobalSize = lBlockCount * cBlockSize;
                    size_t lScanCount = lBlockCount * (1 << cBits) / 4;
                    size_t lScanSize = ceil((float)lScanCount / cBlockSize) * cBlockSize;

                    for (int j = iStartBit; j < iEndBit; j += cBits)
                    {
                        clBlockSort->setArg(0, bfKey);
                        clBlockSort->setArg(1, bfTempKey);
                        clBlockSort->setArg(2, nullptr);
                        clBlockSort->setArg(3, nullptr);
                        clBlockSort->setArg(4, j);
                        clBlockSort->setArg(5, bfBlockScan);
                        clBlockSort->setArg(6, bfBlockOffset);
                        clBlockSort->setArg(7, count);
                        clBlockSort->setArg(8, cBlockSize * sizeof(cl_uint), nullptr);
                        clBlockSort->setArg(9, cBlockSize * sizeof(cl_uint), nullptr);
                        queue->enqueueKernel(clBlockSort, 1, &lGlobalSize, &cBlockSize);
                        queue->finish();

                        clBlockScan->setArg(0, bfBlockScan);
                        clBlockScan->setArg(1, bfBlockSum);
                        clBlockScan->setArg(2, lScanCount);
                        clBlockScan->setArg(3, cBlockSize * sizeof(cl_uint), nullptr);
                        queue->enqueueKernel(clBlockScan, 1, &lScanSize, &cBlockSize);
                        queue->finish();

                        clBlockPrefix->setArg(0, bfBlockScan);
                        clBlockPrefix->setArg(1, bfBlockSum);
                        clBlockPrefix->setArg(2, lScanCount);
                        clBlockPrefix->setArg(3, cBlockSize * sizeof(cl_uint), nullptr);
                        queue->enqueueKernel(clBlockPrefix, 1, &lScanSize, &cBlockSize);
                        queue->finish();

                        clReorder->setArg(0, bfTempKey);
                        clReorder->setArg(1, bfKey);
                        clReorder->setArg(2, nullptr);
                        clReorder->setArg(3, nullptr);
                        clReorder->setArg(4, bfBlockScan);
                        clReorder->setArg(5, bfBlockOffset);
                        clReorder->setArg(6, j);
                        clReorder->setArg(7, count);
                        queue->enqueueKernel(clReorder, 1, &lGlobalSize, &cBlockSize);
                        queue->finish();
                    }

                    queue->finish();
                }

                void download(CommandQueue* queue, T* data) override
                {
                    queue->enqueueRead(bfKey, data);
                    queue->finish();
                }

                void cleanup() override
                {
                    delete program;
                    delete clBlockSort;
                    delete clBlockScan;
                    delete clBlockPrefix;
                    delete clReorder;
                    delete bfKey;
                    delete bfTempKey;
                    //delete bfTempVal;
                    delete bfBlockScan;
                    delete bfBlockSum;
                    delete bfBlockOffset;
                }

                virtual ~RadixSort() {}

            private:
                Program* program;
                Kernel* clBlockSort;
                Kernel* clBlockScan;
                Kernel* clBlockPrefix;
                Kernel* clReorder;
                Buffer* bfKey;
                Buffer* bfTempKey;
                //Buffer* bfTempVal;
                Buffer* bfBlockScan;
                Buffer* bfBlockSum;
                Buffer* bfBlockOffset;
        };
    }
}

#endif // LibCLRadixSort
