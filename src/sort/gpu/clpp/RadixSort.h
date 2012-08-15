#ifndef GPUCLPPRADIXSORT_H
#define GPUCLPPRADIXSORT_H

#include "../../GPUSortingAlgorithm.h"

using namespace std;

#define WORK_GROUP_SIZE 32
#define _bits 32

namespace gpu
{
    namespace clpp
    {
        template<typename T, size_t count>
        class RadixSort : public GPUSortingAlgorithm<T, count>
        {
                using Base = GPUSortingAlgorithm<T, count>;

            public:
                RadixSort(Context* context, CommandQueue* queue)
                    : GPUSortingAlgorithm<T, count>("Radixsort (clpp)", context, queue)
                {
                }

                virtual ~RadixSort()
                {
                }

            protected:
                bool init()
                {
                    program = Base::context->createProgram("gpu/clpp/RadixSort.cl");
                    _kernel_LocalHistogram = program->createKernel("kernel__localHistogram");
                    _kernel_RadixLocalSort = program->createKernel("kernel__radixLocalSort");
                    scanKernel = program->createKernel("kernel__scan_block_anylength");
                    _kernel_RadixPermute = program->createKernel("kernel__radixPermute");

                    scanWorkGroupSize = scanKernel->getWorkGroupSize();

                    return true;
                }

                void upload()
                {
                    unsigned int numBlocks = roundUpDiv(count, WORK_GROUP_SIZE * 4);

                    _clBuffer_radixHist1 = Base::context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * 16 * numBlocks, nullptr);
                    //_clBuffer_radixHist1 = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, _keySize * 16 * numBlocks, NULL, &clStatus);

                    _clBuffer_radixHist2 = Base::context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * 16 * numBlocks, nullptr);
                    //_clBuffer_radixHist2 = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, _keySize * 2 * 16 * numBlocks, NULL, &clStatus);

                    _clBuffer_dataSet = Base::context->createBuffer(CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(T) * count, SortingAlgorithm<T, count>::data);
                    //_clBuffer_dataSet = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(T) * count, _dataSet);

                    _clBuffer_dataSetOut = Base::context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * count, nullptr);
                    //_clBuffer_dataSetOut = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, _keySize * _datasetSize, NULL, &clStatus);

                }

                inline int roundUpDiv(int A, int B)
                {
                    return (A + B - 1) / (B);
                }

                size_t toMultipleOf(size_t N, size_t base)
                {
                    return (ceil((double)N / (double)base) * base);
                }

                void sort(size_t workGroupSize)
                {
                    // Satish et al. empirically set b = 4. The size of a work-group is in hundreds of
                    // work-items, depending on the concrete device and each work-item processes more than one
                    // stream element, usually 4, in order to hide latencies.

                    workGroupSize = WORK_GROUP_SIZE;

                    unsigned int numBlocks = roundUpDiv(count, workGroupSize * 4);
                    unsigned int Ndiv4 = roundUpDiv(count, 4);

                    size_t global[1] = {toMultipleOf(Ndiv4, workGroupSize)};
                    size_t local[1] = {workGroupSize};

                    Buffer* dataA = _clBuffer_dataSet;
                    Buffer* dataB = _clBuffer_dataSetOut;
                    for(unsigned int bitOffset = 0; bitOffset < _bits; bitOffset += 4)
                    {
                        // 1) Each workgroup sorts its tile by using local memory
                        // 2) Create an histogram of d=2^b digits entries

                        radixLocal(global, local, dataA, _clBuffer_radixHist1, _clBuffer_radixHist2, bitOffset);

                        localHistogram(global, local, dataA, _clBuffer_radixHist1, _clBuffer_radixHist2, bitOffset);

                        scan();

                        radixPermute(global, local, dataA, dataB, _clBuffer_radixHist1, _clBuffer_radixHist2, bitOffset, numBlocks);

                        std::swap(dataA, dataB);
                    }

                    Base::queue->finish();
                }

                void radixLocal(const size_t* global, const size_t* local, Buffer* data, Buffer* hist, Buffer* blockHists, int bitOffset)
                {
                    size_t workgroupSize = WORK_GROUP_SIZE * 4;

                    unsigned int Ndiv = roundUpDiv(count, 4); // Each work item handle 4 entries
                    size_t global_128[1] = {toMultipleOf(Ndiv, workgroupSize)};
                    size_t local_128[1] = {workgroupSize};

                    /*if (_keysOnly)
                    	clStatus  = clSetKernelArg(_kernel_RadixLocalSort, a++, _keySize * 2 * 4 * workgroupSize, (const void*)NULL);
                    else
                    	clStatus  = clSetKernelArg(_kernel_RadixLocalSort, a++, (_valueSize+_keySize) * 2 * 4 * workgroupSize, (const void*)NULL);// 2 KV array of 128 items (2 for permutations)*/
                    _kernel_RadixLocalSort->setArg(0, data);
                    _kernel_RadixLocalSort->setArg(1, bitOffset);
                    _kernel_RadixLocalSort->setArg(2, count);

                    //clStatus |= clSetKernelArg(_kernel_RadixLocalSort, a++, sizeof(cl_mem), (const void*)&data);
                    //clStatus |= clSetKernelArg(_kernel_RadixLocalSort, a++, sizeof(int), (const void*)&bitOffset);
                    //clStatus |= clSetKernelArg(_kernel_RadixLocalSort, a++, sizeof(unsigned int), (const void*)&_datasetSize);
                    //clStatus |= clEnqueueNDRangeKernel(_context->clQueue, _kernel_RadixLocalSort, 1, NULL, global_128, local_128, 0, NULL, NULL);

                    Base::queue->enqueueKernel(_kernel_RadixLocalSort, 1, global_128, local_128);
                }

                void localHistogram(const size_t* global, const size_t* local, Buffer* data, Buffer* hist, Buffer* blockHists, int bitOffset)
                {
                    /*cl_int clStatus;
                    clStatus = clSetKernelArg(_kernel_LocalHistogram, 0, sizeof(cl_mem), (const void*)&data);
                    clStatus |= clSetKernelArg(_kernel_LocalHistogram, 1, sizeof(int), (const void*)&bitOffset);
                    clStatus |= clSetKernelArg(_kernel_LocalHistogram, 2, sizeof(cl_mem), (const void*)&hist);
                    clStatus |= clSetKernelArg(_kernel_LocalHistogram, 3, sizeof(cl_mem), (const void*)&blockHists);
                    clStatus |= clSetKernelArg(_kernel_LocalHistogram, 4, sizeof(unsigned int), (const void*)&_datasetSize);
                    clStatus |= clEnqueueNDRangeKernel(_context->clQueue, _kernel_LocalHistogram, 1, NULL, global, local, 0, NULL, NULL);*/

                    _kernel_LocalHistogram->setArg(0, data);
                    _kernel_LocalHistogram->setArg(1, bitOffset);
                    _kernel_LocalHistogram->setArg(2, hist);
                    _kernel_LocalHistogram->setArg(3, blockHists);
                    _kernel_LocalHistogram->setArg(4, count);
                    Base::queue->enqueueKernel(_kernel_LocalHistogram, 1, global, local);
                }

                void scan()
                {
                    size_t size = _clBuffer_radixHist1->getSize();

                    int blockSize = size / scanWorkGroupSize;
                    int B = blockSize * scanWorkGroupSize;
                    if ((size % scanWorkGroupSize) > 0)
                        blockSize++;

                    size_t localWorkSize[] = {scanWorkGroupSize};
                    size_t globalWorkSize[] = {toMultipleOf(size / blockSize, scanWorkGroupSize)};

                    /*clStatus  = clSetKernelArg(kernel__scan, 0, scanWorkGroupSize * sizeof(T), 0);
                    clStatus |= clSetKernelArg(kernel__scan, 1, sizeof(cl_mem), &_clBuffer_values);
                    clStatus |= clSetKernelArg(kernel__scan, 2, sizeof(int), &B);
                    clStatus |= clSetKernelArg(kernel__scan, 3, sizeof(int), &size);
                    clStatus |= clSetKernelArg(kernel__scan, 4, sizeof(int), &blockSize);
                    clStatus |= clEnqueueNDRangeKernel(_context->clQueue, kernel__scan, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);*/

                    scanKernel->setArg(0, scanWorkGroupSize * sizeof(T), nullptr);
                    scanKernel->setArg(1, _clBuffer_radixHist1);
                    scanKernel->setArg(2, B);
                    scanKernel->setArg(3, size);
                    scanKernel->setArg(4, blockSize);
                    Base::queue->enqueueKernel(scanKernel, 1, globalWorkSize, localWorkSize);
                }

                void radixPermute(const size_t* global, const size_t* local, Buffer* dataIn, Buffer* dataOut, Buffer* histScan, Buffer* blockHists, int bitOffset, unsigned int numBlocks)
                {
                    /*cl_int clStatus;
                    clStatus  = clSetKernelArg(_kernel_RadixPermute, 0, sizeof(cl_mem), (const void*)&dataIn);
                    clStatus |= clSetKernelArg(_kernel_RadixPermute, 1, sizeof(cl_mem), (const void*)&dataOut);
                    clStatus |= clSetKernelArg(_kernel_RadixPermute, 2, sizeof(cl_mem), (const void*)&histScan);
                    clStatus |= clSetKernelArg(_kernel_RadixPermute, 3, sizeof(cl_mem), (const void*)&blockHists);
                    clStatus |= clSetKernelArg(_kernel_RadixPermute, 4, sizeof(int), (const void*)&bitOffset);
                    clStatus |= clSetKernelArg(_kernel_RadixPermute, 5, sizeof(unsigned int), (const void*)&_datasetSize);
                    clStatus |= clSetKernelArg(_kernel_RadixPermute, 6, sizeof(unsigned int), (const void*)&numBlocks);
                    clStatus |= clEnqueueNDRangeKernel(_context->clQueue, _kernel_RadixPermute, 1, NULL, global, local, 0, NULL, NULL);*/

                    _kernel_RadixPermute->setArg(0, dataIn);
                    _kernel_RadixPermute->setArg(1, dataOut);
                    _kernel_RadixPermute->setArg(2, histScan);
                    _kernel_RadixPermute->setArg(3, blockHists);
                    _kernel_RadixPermute->setArg(4, bitOffset);
                    _kernel_RadixPermute->setArg(5, count);
                    _kernel_RadixPermute->setArg(6, numBlocks);
                    Base::queue->enqueueKernel(_kernel_RadixPermute, 1, global, local);
                }



                void download()
                {
                    // clStatus = clEnqueueReadBuffer(_context->clQueue, _clBuffer_dataSetOut, CL_TRUE, 0, _keySize * _datasetSize, _dataSetOut, 0, NULL, NULL);
                    Base::queue->enqueueRead(_clBuffer_dataSetOut, SortingAlgorithm<T, count>::data);
                    Base::queue->finish();
                }

                void cleanup()
                {
                    delete program;
                    delete _clBuffer_dataSet;
                    delete _clBuffer_dataSetOut;
                    delete _clBuffer_radixHist1;
                    delete _clBuffer_radixHist2;
                    delete _kernel_RadixLocalSort;
                    delete _kernel_LocalHistogram;
                    delete scanKernel;
                    delete _kernel_RadixPermute;
                }

                size_t scanWorkGroupSize;

                Program* program;
                Kernel* _kernel_RadixLocalSort;
                Kernel* _kernel_LocalHistogram;
                Kernel* scanKernel;
                Kernel* _kernel_RadixPermute;
                Buffer* _clBuffer_dataSet;
                Buffer* _clBuffer_dataSetOut;
                Buffer* _clBuffer_radixHist1;
                Buffer* _clBuffer_radixHist2;
        };
    }
}

#endif // GPUCLPPRADIXSORT_H
