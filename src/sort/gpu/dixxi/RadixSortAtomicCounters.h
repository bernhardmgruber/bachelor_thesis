#pragma once

#include "../../../common/CLAlgorithm.h"
#include "../../SortAlgorithm.h"

namespace gpu
{
    namespace dixxi
    {
        template<typename T>
        class RadixSortAtomicCounters : public CLAlgorithm<T>, public SortAlgorithm
        {
        public:
            const static size_t RADIX = 3; // there must be at least 8 counters available
            const static size_t BUCKETS = 1 << RADIX;

            const string getName() override
            {
                return "Radixsort Atomic Counters (dixxi)";
            }

            bool isInPlace() override
            {
                return false;
            }

            void init() override
            {
                stringstream ss;
                ss << "-D T=" << getTypeName<T>() << " -D RADIX=" << RADIX;
                Program* program = context->createProgram("gpu/dixxi/RadixSortAtomicCounters.cl", ss.str());

                histogramKernel = program->createKernel("Histogram");
                scanKernel = program->createKernel("Scan");
                permuteKernel = program->createKernel("Permute");

                delete program;

                histogram = context->createBuffer(CL_MEM_READ_WRITE, BUCKETS * sizeof(cl_uint));

                //histograms = new Buffer*[BUCKETS];
                //for(size_t i = 0; i < BUCKETS; i++)
                //    histograms[i] = histogram->createSubBuffer(CL_MEM_READ_WRITE, i * sizeof(cl_uint), sizeof(cl_uint));
            }

            void upload(size_t workGroupSize, T* data, size_t size) override
            {
                src = context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * size);
                queue->enqueueWrite(src, data);
                dest = context->createBuffer(CL_MEM_READ_WRITE, sizeof(T) * size);
            }

            void run(size_t workGroupSize, size_t size) override
            {
                run_r(workGroupSize, 0, (cl_uint)(size - 1), (cl_uint)(sizeof(T) * 8 - RADIX), (cl_uint)size);
            }

            void run_r(size_t workGroupSize, cl_uint start, cl_uint end, cl_uint bits, cl_uint size)
            {
                if(bits > sizeof(T) * 8 - RADIX)
                    return; // handle underflow

                // zero histogram
                queue->enqueueFill(histogram, 0);

                //unsigned int* lol = new unsigned int[size];
                //queue->enqueueRead(src, lol);

                //cout << "Src mem" << endl;
                //printArr(lol, size);

                // calculate histogram
                size_t globalWorkSizes[] = { roundToMultiple(end - start + 1, workGroupSize) };
                size_t localWorkSizes[] = { workGroupSize };

                histogramKernel->setArg(0, src);
                histogramKernel->setArg(1, histogram);
                histogramKernel->setArg(2, start);
                histogramKernel->setArg(3, end);
                histogramKernel->setArg(4, bits);

                queue->enqueueKernel(histogramKernel, 1, globalWorkSizes, localWorkSizes);

                //queue->enqueueRead(histogram, lol);
                //cout << "Histogram" << endl;
                //printArr(lol, BUCKETS);

                // scan the histogram (exclusive scan)
                scanKernel->setArg(0, histogram);
                queue->enqueueTask(scanKernel);

                //queue->enqueueRead(histogram, lol);
                //cout << "Scanned Histogram" << endl;
                //printArr(lol, BUCKETS);

                // copy histogram into counters
                //for(size_t i = 0; i < BUCKETS; i++)
                //    queue->enqueueCopy(histogram, histograms[i], i * sizeof(cl_uint), 0, sizeof(cl_uint));

                //cout << "Histogram before permute" << endl;
                //cl_uint h;
                //for(size_t i = 0; i < BUCKETS; i++)
                //{
                //    queue->enqueueRead(histograms[i], &h);
                //    cout << h << " ";
                //}
                //cout << endl;

                // rearrange the elements based on scaned histogram
                permuteKernel->setArg(0, src);
                permuteKernel->setArg(1, dest);
                permuteKernel->setArg(2, histogram);
                permuteKernel->setArg(3, start);
                permuteKernel->setArg(4, end);
                permuteKernel->setArg(5, bits);

                queue->enqueueKernel(permuteKernel, 1, globalWorkSizes, localWorkSizes);

                /*queue->enqueueRead(src, lol);
                cout << "Src after permute" << endl;
                printArr(lol, size);
                queue->enqueueRead(dest, lol);
                cout << "Dest after permute" << endl;
                printArr(lol, size);*/

                //cout << "Histogram after permute" << endl;
                //for(size_t i = 0; i < BUCKETS; i++)
                //{
                //    queue->enqueueRead(histograms[i], &h);
                //    cout << h << " ";
                //}
                //cout << endl;

                std::swap(src, dest);

                cl_uint hist[BUCKETS + 1];
                hist[BUCKETS] = size;
                queue->enqueueRead(histogram, hist);

                for(int i = 0; i < BUCKETS; i++)
                    run_r(workGroupSize, hist[i], hist[i + 1] - 1, bits >> RADIX, size);
            }

            void download(T* result, size_t size) override
            {
                queue->enqueueRead(src, result);

                delete src;
                delete dest;
            }

            void cleanup() override
            {
                delete histogramKernel;
                delete scanKernel;
                delete permuteKernel;
                //for(size_t i = 0; i < BUCKETS; i++)
                //    delete histograms[i];
                //delete[] histograms;
                delete histogram;
            }

            virtual ~RadixSortAtomicCounters() {}

        private:
            Kernel* histogramKernel;
            Kernel* scanKernel;
            Kernel* permuteKernel;
            Buffer* src;
            Buffer* dest;
            Buffer* histogram;
            Buffer** histograms;
        };
    }
}
