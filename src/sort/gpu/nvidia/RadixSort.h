#ifndef GPUNVIDIARADIXSORT_H
#define GPUNVIDIARADIXSORT_H

#include "../../../common/GPUAlgorithm.h"
#include "../../SortAlgorithm.h"

// Scan
#define MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE 1024
#define MAX_LOCAL_GROUP_SIZE 256

namespace gpu
{
    namespace nvidia
    {
        template<typename T>
        class RadixSort : public GPUAlgorithm<T>, public SortAlgorithm
        {
            public:
                // Radixsort
                static const size_t CTA_SIZE = 128;
                static const size_t keyBits = 32;
                static const size_t WARP_SIZE = 32;
                static const size_t bitStep = 4;

                // Scan
                static const int WORKGROUP_SIZE = 256;
                static const size_t MAX_BATCH_ELEMENTS = 64 * 1048576;
                static const size_t MIN_SHORT_ARRAY_SIZE = 4;
                static const size_t MAX_SHORT_ARRAY_SIZE = 4 * WORKGROUP_SIZE;
                static const size_t MIN_LARGE_ARRAY_SIZE = 8 * WORKGROUP_SIZE;
                static const size_t MAX_LARGE_ARRAY_SIZE = 4 * WORKGROUP_SIZE * WORKGROUP_SIZE;

                const string getName() override
                {
                    return "Radixsort (NVIDIA)";
                }

                bool isInPlace() override
                {
                    return false;
                }

                void init(Context* context) override
                {
                    Program* program = context->createProgram("gpu/nvidia/RadixSort.cl", "-D T=" + getTypeName<T>());
                    ckRadixSortBlocksKeysOnly = program->createKernel("radixSortBlocksKeysOnly");
                    ckFindRadixOffsets = program->createKernel("findRadixOffsets");
                    ckScanNaive = program->createKernel("scanNaive");
                    ckReorderDataKeysOnly = program->createKernel("reorderDataKeysOnly");
                    delete program;

                    // Scan program
                    program = context->createProgram("gpu/nvidia/Scan_b.cl", "-D T=" + getTypeName<T>() + " -cl-fast-relaxed-math");
                    ckScanExclusiveLocal1 = program->createKernel("scanExclusiveLocal1");
                    ckScanExclusiveLocal2 = program->createKernel("scanExclusiveLocal2");
                    ckUniformUpdate = program->createKernel("uniformUpdate");
                    delete program;
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
                {
                    bufferSize = pow2roundup(size);

                    size_t maxElements = bufferSize;
                    size_t numBlocks = ((maxElements % (CTA_SIZE * 4)) == 0) ? (maxElements / (CTA_SIZE * 4)) : (maxElements / (CTA_SIZE * 4) + 1);

                    d_tempKeys = context->createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint) * maxElements);
                    mCounters = context->createBuffer(CL_MEM_READ_WRITE, WARP_SIZE * numBlocks * sizeof(T));
                    mCountersSum = context->createBuffer(CL_MEM_READ_WRITE, WARP_SIZE * numBlocks * sizeof(T));
                    mBlockOffsets = context->createBuffer(CL_MEM_READ_WRITE, WARP_SIZE * numBlocks * sizeof(T));

                    size_t numElementsForScan = maxElements/2/CTA_SIZE*16;
                    d_Buffer = context->createBuffer(CL_MEM_READ_WRITE, numElementsForScan / MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE * sizeof(T));

                    d_keys = context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * bufferSize);
                    queue->enqueueWrite(d_keys, data, 0, size * sizeof(T));
                }

                void radixSortBlocksKeysOnlyOCL(CommandQueue* queue, Buffer* d_keys, unsigned int nbits, unsigned int startbit, unsigned int numElements)
                {
                    unsigned int totalBlocks = numElements/4/CTA_SIZE;
                    size_t globalWorkSize[] = {CTA_SIZE*totalBlocks};
                    size_t localWorkSize[] = {CTA_SIZE};

                    ckRadixSortBlocksKeysOnly->setArg(0, d_keys);
                    ckRadixSortBlocksKeysOnly->setArg(1, d_tempKeys);
                    ckRadixSortBlocksKeysOnly->setArg(2, (cl_uint)nbits);
                    ckRadixSortBlocksKeysOnly->setArg(3, (cl_uint)startbit);
                    ckRadixSortBlocksKeysOnly->setArg(4, (cl_uint)numElements);
                    ckRadixSortBlocksKeysOnly->setArg(5, (cl_uint)totalBlocks);
                    ckRadixSortBlocksKeysOnly->setArg(6, 4*CTA_SIZE*sizeof(T), nullptr);

                    queue->enqueueKernel(ckRadixSortBlocksKeysOnly, 1, globalWorkSize, localWorkSize);
                }

                void findRadixOffsetsOCL(CommandQueue* queue, unsigned int startbit, unsigned int numElements)
                {
                    unsigned int totalBlocks = numElements/2/CTA_SIZE;
                    size_t globalWorkSize[1] = {CTA_SIZE*totalBlocks};
                    size_t localWorkSize[1] = {CTA_SIZE};


                    ckFindRadixOffsets->setArg(0, d_tempKeys);
                    ckFindRadixOffsets->setArg(1, mCounters);
                    ckFindRadixOffsets->setArg(2, mBlockOffsets);
                    ckFindRadixOffsets->setArg(3, (cl_uint)startbit);
                    ckFindRadixOffsets->setArg(4, (cl_uint)numElements);
                    ckFindRadixOffsets->setArg(5, (cl_uint)totalBlocks);
                    ckFindRadixOffsets->setArg(6, 2 * CTA_SIZE *sizeof(T), nullptr);

                    queue->enqueueKernel(ckFindRadixOffsets, 1, globalWorkSize, localWorkSize);
                }

                /*#define NUM_BANKS 16
                                void RadixSort::scanNaiveOCL(unsigned int numElements)
                                {
                                    unsigned int nHist = numElements/2/CTA_SIZE*16;
                                    size_t globalWorkSize[1] = {nHist};
                                    size_t localWorkSize[1] = {nHist};
                                    unsigned int extra_space = nHist / NUM_BANKS;
                                    unsigned int shared_mem_size = sizeof(unsigned int) * (nHist + extra_space);
                                    cl_int ciErrNum;
                                    ciErrNum  = clSetKernelArg(ckScanNaive, 0, sizeof(cl_mem), (void*)&mCountersSum);
                                    ciErrNum |= clSetKernelArg(ckScanNaive, 1, sizeof(cl_mem), (void*)&mCounters);
                                    ciErrNum |= clSetKernelArg(ckScanNaive, 2, sizeof(unsigned int), (void*)&nHist);
                                    ciErrNum |= clSetKernelArg(ckScanNaive, 3, 2 * shared_mem_size, NULL);
                                    ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckScanNaive, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
                                    oclCheckError(ciErrNum, CL_SUCCESS);
                                }*/

                void reorderDataKeysOnlyOCL(CommandQueue* queue, Buffer* d_keys, unsigned int startbit, unsigned int numElements)
                {
                    unsigned int totalBlocks = numElements/2/CTA_SIZE;
                    size_t globalWorkSize[1] = {CTA_SIZE*totalBlocks};
                    size_t localWorkSize[1] = {CTA_SIZE};

                    ckReorderDataKeysOnly->setArg(0, d_keys);
                    ckReorderDataKeysOnly->setArg(1, d_tempKeys);
                    ckReorderDataKeysOnly->setArg(2, mBlockOffsets);
                    ckReorderDataKeysOnly->setArg(3, mCountersSum);
                    ckReorderDataKeysOnly->setArg(4, mCounters);
                    ckReorderDataKeysOnly->setArg(5, (cl_uint)startbit);
                    ckReorderDataKeysOnly->setArg(6, (cl_uint)numElements);
                    ckReorderDataKeysOnly->setArg(7, (cl_uint)totalBlocks);
                    ckReorderDataKeysOnly->setArg(8, 2 * CTA_SIZE * sizeof(T), nullptr);

                    queue->enqueueKernel(ckReorderDataKeysOnly, 1, globalWorkSize, localWorkSize);
                }

                void radixSortStepKeysOnly(CommandQueue* queue, Buffer* d_keys, unsigned int nbits, unsigned int startbit, unsigned int numElements)
                {
                    // Four step algorithms from Satish, Harris & Garland
                    radixSortBlocksKeysOnlyOCL(queue, d_keys, nbits, startbit, numElements);

                    findRadixOffsetsOCL(queue, startbit, numElements);

                    scanExclusiveLarge(queue, mCountersSum, mCounters, 1, numElements/2/CTA_SIZE*16);

                    reorderDataKeysOnlyOCL(queue, d_keys, startbit, numElements);
                }

                void run(CommandQueue* queue, size_t workGroupSize, size_t size) override
                {
                    int i = 0;
                    while (keyBits > i * bitStep)
                    {
                        radixSortStepKeysOnly(queue, d_keys, bitStep, i*bitStep, bufferSize);
                        i++;
                    }
                }

                void download(CommandQueue* queue, T* result, size_t size) override
                {
                    queue->enqueueRead(d_keys, result, 0, size * sizeof(T));

                    delete d_tempKeys;
                    delete mCounters;
                    delete mCountersSum;
                    delete mBlockOffsets;

                    delete d_keys;

                    delete d_Buffer;
                }

                void cleanup() override
                {
                    delete ckRadixSortBlocksKeysOnly;
                    delete ckFindRadixOffsets;
                    delete ckScanNaive;
                    delete ckReorderDataKeysOnly;
                }

                virtual ~RadixSort() {}

            private:
                void scanExclusiveLarge(CommandQueue* queue, Buffer* d_Dst, Buffer* d_Src, unsigned int batchSize, unsigned int arrayLength)
                {
                    //Check supported size range
                    if((arrayLength < MIN_LARGE_ARRAY_SIZE) || (arrayLength > MAX_LARGE_ARRAY_SIZE) )
                        throw OpenCLException("Element count is out of supported size");

                    //Check total batch size limit
                    if((batchSize * arrayLength) > MAX_BATCH_ELEMENTS)
                        throw OpenCLException("Batch element count is out of supported size");

                    scanExclusiveLocal1(queue, d_Dst, d_Src, (batchSize * arrayLength) / (4 * WORKGROUP_SIZE), 4 * WORKGROUP_SIZE);

                    scanExclusiveLocal2(queue, d_Buffer, d_Dst, d_Src, batchSize, arrayLength / (4 * WORKGROUP_SIZE));

                    uniformUpdate(queue, d_Dst, d_Buffer, (batchSize * arrayLength) / (4 * WORKGROUP_SIZE));
                }

                void scanExclusiveLocal1(CommandQueue* queue, Buffer* d_Dst, Buffer* d_Src, unsigned int n, unsigned int size)
                {
                    ckScanExclusiveLocal1->setArg(0, d_Dst);
                    ckScanExclusiveLocal1->setArg(1, d_Src);
                    ckScanExclusiveLocal1->setArg(2, 2 * WORKGROUP_SIZE * sizeof(T), nullptr);
                    ckScanExclusiveLocal1->setArg(3, (cl_uint)size);

                    size_t localWorkSize[] = { WORKGROUP_SIZE };
                    size_t globalWorkSize[] = { (n * size) / 4 };

                    queue->enqueueKernel(ckScanExclusiveLocal1, 1, globalWorkSize, localWorkSize);
                }

                void scanExclusiveLocal2(CommandQueue* queue, Buffer* d_Buffer, Buffer* d_Dst, Buffer* d_Src, unsigned int n, unsigned int size)
                {
                    unsigned int elements = n * size;

                    ckScanExclusiveLocal2->setArg(0, d_Buffer);
                    ckScanExclusiveLocal2->setArg(1, d_Dst);
                    ckScanExclusiveLocal2->setArg(2, d_Src);
                    ckScanExclusiveLocal2->setArg(3, 2 * WORKGROUP_SIZE * sizeof(T), nullptr);
                    ckScanExclusiveLocal2->setArg(4, (cl_uint)elements);
                    ckScanExclusiveLocal2->setArg(5, (cl_uint)size);

                    size_t localWorkSize[] = { WORKGROUP_SIZE };
                    size_t globalWorkSize[] = { roundToMultiple(elements, WORKGROUP_SIZE) };

                    queue->enqueueKernel(ckScanExclusiveLocal2, 1, globalWorkSize, localWorkSize);
                }

                void uniformUpdate(CommandQueue* queue, Buffer* d_Dst, Buffer* d_Buffer, unsigned int n)
                {
                    ckUniformUpdate->setArg(0, d_Dst);
                    ckUniformUpdate->setArg(1, d_Buffer);

                    size_t localWorkSize[] = { WORKGROUP_SIZE };
                    size_t globalWorkSize[] = { n * WORKGROUP_SIZE };

                    queue->enqueueKernel(ckUniformUpdate, 1, globalWorkSize, localWorkSize);
                }


                size_t bufferSize;

                Buffer* d_tempKeys;
                Buffer* mCounters;
                Buffer* mCountersSum;
                Buffer* mBlockOffsets;
                Buffer* d_keys;

                Kernel* ckRadixSortBlocksKeysOnly;
                Kernel* ckFindRadixOffsets;
                Kernel* ckScanNaive;
                Kernel* ckReorderDataKeysOnly;

                // Scan
                Kernel* ckScanExclusiveLocal1;
                Kernel* ckScanExclusiveLocal2;
                Kernel* ckUniformUpdate;

                Buffer* d_Buffer;
        };
    }
}

#endif // GPUNVIDIARADIXSORT_H
