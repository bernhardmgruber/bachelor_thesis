#ifndef GPUDIXXIRADIXSORTATOMICCOUNTERS_H
#define GPUDIXXIRADIXSORTATOMICCOUNTERS_H

#include "../../../common/GPUAlgorithm.h"
#include "../../SortAlgorithm.h"

namespace gpu
{
    namespace dixxi
    {
        template<typename T>
        class RadixSortAtomicCounters : public GPUAlgorithm<T>, public SortAlgorithm
        {
            public:
                const static size_t RADIX = 3; // there must be at least 8 counters available
                const static size_t BUCKETS = 1<<RADIX;

                const string getName() override
                {
                    return "Radixsort Atomic Counters (dixxi)";
                }

                bool isInPlace() override
                {
                    return false;
                }

                void init(Context* context) override
                {
                    stringstream ss;
                    ss << "-D T=" << getTypeName<T>() << " -D BUCKETS=" << BUCKETS;
                    Program* program = context->createProgram("gpu/dixxi/RadixSortAtomicCounters.cl", ss.str());

                    histogramKernel = program->createKernel("Histogram");
                    scanKernel = program->createKernel("Scan");
                    permuteKernel = program->createKernel("Permute");
                    zeroHistogramKernel = program->createKernel("ZeroHistogram");

                    delete program;

                    histograms = new Buffer*[BUCKETS];
                    for(size_t i = 0; i < BUCKETS; i++)
                        histograms[i] = context->createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint));
                    histogram = context->createBuffer(CL_MEM_READ_WRITE, BUCKETS * sizeof(cl_uint));
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
                {
                    src = context->createBuffer(CL_MEM_READ_ONLY, sizeof(T) * size);
                    queue->enqueueWrite(src, data);
                    dest = context->createBuffer(CL_MEM_WRITE_ONLY, sizeof(T) * size);
                }

                void run(CommandQueue* queue, size_t workGroupSize, size_t size) override
                {
                    for(size_t bits = 0; bits < sizeof(T) * 8 ; bits += RADIX)
                    {
                        // zero histogram
                        size_t globalWorkSizes[] = { BUCKETS };
                        size_t localWorkSizes[] = { BUCKETS };
                        assert(BUCKETS <= workGroupSize);
                        zeroHistogramKernel->setArg(0, histogram);
                        queue->enqueueKernel(zeroHistogramKernel, 1, globalWorkSizes, localWorkSizes);
                        queue->enqueueBarrier();

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

                        // scan the histogram (exclusive scan)
                        scanKernel->setArg(0, histogram);
                        queue->enqueueTask(scanKernel);
                        queue->enqueueBarrier();

                        queue->enqueueRead(histogram, lol);
                        cout << "Scanned Histogram" << endl;
                        printArr(lol, BUCKETS);

                        // copy histogram into counters
                        for(size_t i = 0; i < BUCKETS; i++)
                            queue->enqueueCopy(histogram, histograms[i], i * sizeof(cl_uint), 0, sizeof(cl_uint));

                        // rearrange the elements based on scaned histogram
                        globalWorkSizes[0] = size;
                        localWorkSizes[0] = workGroupSize;
                        permuteKernel->setArg(0, src);
                        permuteKernel->setArg(1, dest);
                        permuteKernel->setArg(2, bits);
                        for(size_t i = 0; i < BUCKETS; i++)
                            permuteKernel->setArg(3 + i, histograms[i]);
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

                void download(CommandQueue* queue, T* result, size_t size) override
                {
                    queue->enqueueRead(dest, result);
                    delete dest;
                }

                void cleanup() override
                {
                    delete histogramKernel;
                    delete scanKernel;
                    delete permuteKernel;
                    delete zeroHistogramKernel;
                    for(size_t i = 0; i < BUCKETS; i++)
                        delete histograms[i];
                    delete[] histograms;
                }

                virtual ~RadixSortAtomicCounters() {}

            private:
                Kernel* histogramKernel;
                Kernel* scanKernel;
                Kernel* permuteKernel;
                Kernel* zeroHistogramKernel;
                Buffer* src;
                Buffer* dest;
                Buffer* histogram;
                Buffer** histograms;
        };
    }
}

#endif // GPUDIXXIRADIXSORTATOMICCOUNTERS_H
