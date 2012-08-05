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

namespace libcl
{

/*
class oclRadixSort : public oclProgram
{
    public:

        oclRadixSort(oclContext& iContext);

        int compile();
        int compute(oclDevice& iDevice, oclBuffer& bfKey, oclBuffer& bfVal, int iStartBit, int iEndBit);

    protected:

        static const int cBits;
        static const size_t cBlockSize;
        static const size_t cMaxArraySize;

        oclKernel clBlockSort;
        oclKernel clBlockScan;
        oclKernel clBlockPrefix;
        oclKernel clReorder;

        oclBuffer bfTempKey;
        oclBuffer bfTempVal;
        oclBuffer bfBlockScan;
        oclBuffer bfBlockSum;
        oclBuffer bfBlockOffset;

        void fit(oclBuffer& iBuffer, size_t iElements) ;

};
*/

#define CBITS 4
#define BLOCK_SIZE 256
#define BLOCK_SIZE_CUBE BLOCK_SIZE*BLOCK_SIZE*BLOCK_SIZE

#include "../../GPUSortingAlgorithm.h"
#include "../../OpenCL.h"

using namespace std;

/**
 * From: http://www.libcl.org/
 */
template<typename T, size_t count>
class RadixSort : public GPUSortingAlgorithm<T, count>
{
        using Base = GPUSortingAlgorithm<T, count>;

    public:
        RadixSort(Context* context, CommandQueue* queue)
            : GPUSortingAlgorithm<T, count>("Radix sort (LibCL)", context, queue)
        {
        }

        virtual ~RadixSort()
        {
        }

    protected:
        bool init()
        {
            program = Base::context->createProgram("gpu/libCL/RadixSort.cl");

            clBlockSort = program->createKernel("clBlockSort");
            clBlockScan = program->createKernel("clBlockScan");
            clBlockPrefix = program->createKernel("clBlockPrefix");
            clReorder = program->createKernel("clReorder");

            size_t cBits = CBITS;
            //size_t cBlockSize = Base::context->getInfoSize(CL_DEVICE_MAX_WORK_GROUP_SIZE);
            size_t cBlockSize = BLOCK_SIZE;

            size_t lBlockCount = ceil((float)count / cBlockSize);
            //size_t lBlockCount = ceil((float)bfKey.count<cl_uint>()/cBlockSize);
            //fit(bfBlockScan, lBlockCount*(1<<cBits));
            //fit(bfBlockOffset, lBlockCount*(1<<cBits));
            //fit(bfBlockSum, cBlockSize);

            //size_t lElementCount = bfKey.count<cl_uint>();
            //fit(bfTempKey, count);
            //fit(bfTempVal, count);

            bfTempKey = Base::context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * count);
            //bfTempVal = Base::context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * count);
            bfBlockScan = Base::context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * lBlockCount * (1 << cBits));
            bfBlockOffset = Base::context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * lBlockCount * (1 << cBits));
            bfBlockSum = Base::context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * cBlockSize);

            return true;
        }

        void upload()
        {
            bfKey = Base::context->createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(T) * count, SortingAlgorithm<T, count>::data);

            Base::queue->finish();
        }

        /*void fit(Buffer* iBuffer, size_t iElements)
        {
            if (iBuffer->getSize() / sizeof(int) < iElements)
            {
                cerr << "NOT IMPLEMENTED" << endl;
                //iBuffer.resize<cl_uint>(iElements);
            }
        }*/

        void sort(size_t workGroupSize)
        {
            int iStartBit = 0;
            int iEndBit = 32;

            size_t cBits = CBITS;
            size_t cBlockSize = BLOCK_SIZE;

            /*if (bfKey.dim(0) != bfVal.dim(0))
            {
                cerr << "key and value arrays are of different size ( " << bfKey.getMemObjectInfo<size_t>(CL_MEM_SIZE) << "," << bfVal.getMemObjectInfo<size_t>(CL_MEM_SIZE) << ")";
                return false;
            }*/

            /*if (bfKey.count<cl_uint>() >= cMaxArraySize)
            {
                cerr << "maximum sortable array size = " << cMaxArraySize;
                return false;
            }*/

            if ((iEndBit - iStartBit) % cBits != 0)
            {
                cerr << "end bit(" << iEndBit << ") - start bit(" << iStartBit << ") must be divisible by 4";
                return;
            }

            size_t lBlockCount = ceil((float)count / cBlockSize);

            size_t lGlobalSize = lBlockCount*cBlockSize;
            size_t lScanCount = lBlockCount*(1<<cBits)/4;
            size_t lScanSize = ceil((float)lScanCount/cBlockSize)*cBlockSize;

            for (int j=iStartBit; j<iEndBit; j+=cBits)
            {
                clBlockSort->setArg(0, bfKey);
                clBlockSort->setArg(1, bfTempKey);
                clBlockSort->setArg(2, nullptr);
                clBlockSort->setArg(3, nullptr);
                clBlockSort->setArg(4, j);
                clBlockSort->setArg(5, bfBlockScan);
                clBlockSort->setArg(6, bfBlockOffset);
                clBlockSort->setArg(7, count);
                /*clSetKernelArg(clBlockSort, 0, sizeof(cl_mem), bfKey);
                clSetKernelArg(clBlockSort, 1, sizeof(cl_mem), bfTempKey);
                clSetKernelArg(clBlockSort, 2, sizeof(cl_mem), bfVal);
                clSetKernelArg(clBlockSort, 3, sizeof(cl_mem), bfTempVal);
                clSetKernelArg(clBlockSort, 4, sizeof(cl_uint), &j);
                clSetKernelArg(clBlockSort, 5, sizeof(cl_mem), bfBlockScan);
                clSetKernelArg(clBlockSort, 6, sizeof(cl_mem), bfBlockOffset);
                clSetKernelArg(clBlockSort, 7, sizeof(cl_uint), &lElementCount);*/
                Base::queue->enqueueKernel(clBlockSort, 1, &lGlobalSize, &cBlockSize);
                //sStatusCL = clEnqueueNDRangeKernel(queue, clBlockSort, 1, NULL, &lGlobalSize, &cBlockSize, 0, NULL, clBlockSort.getEvent());
                /*if (!oclSuccess("clEnqueueNDRangeKernel", this))
                {
                    return false;
                }*/

                clBlockScan->setArg(0, bfBlockScan);
                clBlockScan->setArg(1, bfBlockSum);
                clBlockScan->setArg(2, lScanCount);
                Base::queue->enqueueKernel(clBlockScan, 1, &lScanSize, &cBlockSize);

                //lSetKernelArg(clBlockScan, 0, sizeof(cl_mem), bfBlockScan);
                //clSetKernelArg(clBlockScan, 1, sizeof(cl_mem), bfBlockSum);
                //clSetKernelArg(clBlockScan, 2, sizeof(cl_uint), &lScanCount);
                //sStatusCL = clEnqueueNDRangeKernel(queue, clBlockScan, 1, NULL, &lScanSize, &cBlockSize, 0, NULL, clBlockScan.getEvent());
                /*if (!oclSuccess("clEnqueueNDRangeKernel", this))
                {
                    return false;
                }*/


                clBlockPrefix->setArg(0, bfBlockScan);
                clBlockPrefix->setArg(1, bfBlockSum);
                clBlockPrefix->setArg(2, lScanCount);
                //clSetKernelArg(clBlockPrefix, 0, sizeof(cl_mem), bfBlockScan);
                //clSetKernelArg(clBlockPrefix, 1, sizeof(cl_mem), bfBlockSum);
                Base::queue->enqueueKernel(clBlockPrefix, 1, &lScanSize, &cBlockSize);
                //clSetKernelArg(clBlockPrefix, 2, sizeof(cl_uint), &lScanCount);
                /*sStatusCL = clEnqueueNDRangeKernel(queue, clBlockPrefix, 1, NULL, &lScanSize, &cBlockSize, 0, NULL, clBlockPrefix.getEvent());
                if (!oclSuccess("clEnqueueNDRangeKernel", this))
                {
                    return false;
                }*/

                clReorder->setArg(0, bfTempKey);
                clReorder->setArg(1, bfKey);
                clReorder->setArg(2, nullptr);
                clReorder->setArg(3, nullptr);
                clReorder->setArg(4, bfBlockScan);
                clReorder->setArg(5, bfBlockOffset);
                clReorder->setArg(6, j);
                clReorder->setArg(7, count);
                /*clSetKernelArg(clReorder, 0, sizeof(cl_mem), bfTempKey);
                clSetKernelArg(clReorder, 1, sizeof(cl_mem), bfKey);
                clSetKernelArg(clReorder, 2, sizeof(cl_mem), bfTempVal);
                clSetKernelArg(clReorder, 3, sizeof(cl_mem), bfVal);
                clSetKernelArg(clReorder, 4, sizeof(cl_mem), bfBlockScan);
                clSetKernelArg(clReorder, 5, sizeof(cl_mem), bfBlockOffset);
                clSetKernelArg(clReorder, 6, sizeof(cl_uint), &j);
                clSetKernelArg(clReorder, 7, sizeof(cl_uint), &lElementCount);*/
                Base::queue->enqueueKernel(clReorder, 1, &lGlobalSize, &cBlockSize);
                /*sStatusCL = clEnqueueNDRangeKernel(queue, clReorder, 1, NULL, &lGlobalSize, &cBlockSize, 0, NULL, clReorder.getEvent());
                if (!oclSuccess("clEnqueueNDRangeKernel", this))
                {
                    return false;
                }*/
            }
            //return true;

            Base::queue->finish();
        }

        void download()
        {
            Base::queue->enqueueRead(bfKey, SortingAlgorithm<T, count>::data);
            Base::queue->finish();
        }

        void cleanup()
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

#endif // LibCLRadixSort
