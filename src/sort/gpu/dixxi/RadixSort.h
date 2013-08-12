#pragma once

#include "../../../common/CLAlgorithm.h"
#include "../../SortAlgorithm.h"

namespace gpu
{
    namespace dixxi
    {
        template<typename T>
        class RadixSort : public CLAlgorithm<T>, public SortAlgorithm
        {
            public:
                static const size_t RADIX = 4;
                static const size_t BUCKETS = 1<<RADIX;

                const string getName() override
                {
                    return "Radixsort (dixxi)";
                }

                bool isInPlace() override
                {
                    return false;
                }

                void init() override
                {
                    cout << context->getInfo<cl_uint>(CL_DEVICE_MAX_ATOMIC_COUNTERS_EXT) << endl;

                    stringstream ss;
                    ss << "-D T=" << getTypeName<T>() << " -D BUCKETS=" << BUCKETS;
                    Program* program = context->createProgram("gpu/dixxi/RadixSort.cl", ss.str());

                    histogramKernel = program->createKernel("Histogram");
                    scanKernel = program->createKernel("Scan");
                    permuteKernel = program->createKernel("Permute");
                    zeroHistogramKernel = program->createKernel("ZeroHistogram");

                    delete program;

                    histogram = context->createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint) * BUCKETS);
                }

                void upload(size_t workGroupSize, T* data, size_t size) override
                {
                    src = context->createBuffer(CL_MEM_READ_ONLY, sizeof(T) * size);
                    queue->enqueueWrite(src, data);
                    dest = context->createBuffer(CL_MEM_WRITE_ONLY, sizeof(T) * size);
                }

                void run(size_t workGroupSize, size_t size) override
                {
                    for(size_t bits = 0; bits < sizeof(T) * 8 ; bits += RADIX)
                    {
                        // zero histogram
                        cl_uint zero = 0;
                        queue->enqueueFill(histogram, zero);

                        unsigned int* lol = new unsigned int[size];
                        queue->enqueueRead(src, lol);

                        cout << "Src mem" << endl;
                        printArr(lol, size);

                        // calculate histogram
                        globalWorkSizes[0] = size;
                        localWorkSizes[0] = workGroupSize;
                        histogramKernel->setArg(0, src);
                        histogramKernel->setArg(1, histogram);
                        histogramKernel->setArg(2, sizeof(cl_uint) * BUCKETS, nullptr);
                        histogramKernel->setArg(3, bits);
                        queue->enqueueKernel(histogramKernel, 1, globalWorkSizes, localWorkSizes);
                        queue->enqueueBarrier();

                        queue->enqueueRead(histogram, lol);
                        cout << "Histogram" << endl;
                        printArr(lol, BUCKETS);

                        // Scan the histogram (exclusive scan)
                        scanKernel->setArg(0, histogram);
                        queue->enqueueTask(scanKernel);
                        queue->enqueueBarrier();

                        queue->enqueueRead(histogram, lol);
                        cout << "Scanned Histogram" << endl;
                        printArr(lol, BUCKETS);

                        // Rearrange the elements based on scaned histogram
                        globalWorkSizes[0] = size;
                        localWorkSizes[0] = workGroupSize;
                        permuteKernel->setArg(0, src);
                        permuteKernel->setArg(1, dest);
                        permuteKernel->setArg(2, histogram);
                        permuteKernel->setArg(3, bits);
                        queue->enqueueKernel(permuteKernel, 1, globalWorkSizes, localWorkSizes);
                        queue->enqueueBarrier();

                        queue->enqueueRead(src, lol);
                        cout << "Src after permute" << endl;
                        printArr(lol, size);
                        queue->enqueueRead(dest, lol);
                        cout << "Dest after permute" << endl;
                        printArr(lol, size);

                        queue->enqueueCopy(dest, src);
                    }
                }

                void download(T* result, size_t size) override
                {
                    queue->enqueueRead(dest, result);
                    delete src;
                    delete dest;
                }

                void cleanup() override
                {
                    delete histogramKernel;
                    delete scanKernel;
                    delete permuteKernel;
                    delete zeroHistogramKernel;
                    delete histogram;
                }

                virtual ~RadixSort() {}

            private:
                Kernel* histogramKernel;
                Kernel* scanKernel;
                Kernel* permuteKernel;
                Kernel* zeroHistogramKernel;
                Buffer* src;
                Buffer* dest;
                Buffer* histogram;
        };
    }
}
