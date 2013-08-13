#pragma once

#include "../../../common/CLAlgorithm.h"
#include "../../SortAlgorithm.h"

using namespace std;

namespace gpu
{
    namespace amd_dixxi
    {
        /**
        * From: http://developer.amd.com/tools/hc/AMDAPPSDK/samples/Pages/default.aspx
        * Modified algorithm by Bernhard Manfred Gruber.
        */
        template<typename T>
        class RadixSort : public CLAlgorithm<T>, public SortAlgorithm
        {
            static const unsigned int RADIX = 4;
            static const unsigned int BUCKETS = (1 << RADIX);
            static const unsigned int BLOCK_SIZE = 16;

        public:
            const string getName() override
            {
                return "Radix sort (AMD/dixxi)";
            }

            bool isInPlace() override
            {
                return false;
            }

            void init() override
            {
                stringstream ss;
                ss << "-D T=" << getTypeName<T>() << " -D RADIX=" << RADIX << " -D BLOCK_SIZE=" << BLOCK_SIZE;
                Program* program = context->createProgram("gpu/amd_dixxi/RadixSort.cl", ss.str());
                histogramKernel = program->createKernel("histogram");
                permuteKernel = program->createKernel("permute");
                delete program;

                program = context->createProgram("gpu/amd_dixxi/RecursiveScan.cl");
                scanKernel = program->createKernel("WorkEfficientScan");
                addKernel = program->createKernel("AddSums");
                delete program;
            }

            void upload(size_t workGroupSize, T* data, size_t size) override
            {
                bufferSize = roundToMultiple(size, workGroupSize * BLOCK_SIZE);

                srcBuffer = context->createBuffer(CL_MEM_READ_WRITE, bufferSize * sizeof(T));
                dstBuffer = context->createBuffer(CL_MEM_READ_WRITE, bufferSize * sizeof(T));

                if(bufferSize != size)
                {
                    queue->enqueueWrite(srcBuffer, data, 0, size * sizeof(T));
                    queue->enqueueFill(srcBuffer, numeric_limits<T>::max(), size * sizeof(T), (bufferSize - size) * sizeof(T));
                }
                else
                    queue->enqueueWrite(srcBuffer, data);

                // each thread has it's own histogram
                histogramSize = (bufferSize / BLOCK_SIZE) * BUCKETS;
                histogramSize = roundToMultiple(histogramSize, workGroupSize * 2);

                histogramBuffer = context->createBuffer(CL_MEM_READ_WRITE, histogramSize * sizeof(cl_uint));
            }

            void run(size_t workGroupSize, size_t size) override
            {
                size_t localSize = (workGroupSize * BUCKETS * sizeof(cl_ushort));

                size_t globalWorkSizes[] = { bufferSize / BLOCK_SIZE };
                size_t localWorkSizes[] = { workGroupSize };

                for(cl_uint bits = 0; bits < sizeof(T) * 8; bits += RADIX)
                {
                    // Calculate thread-histograms
                    histogramKernel->setArg(0, srcBuffer);
                    histogramKernel->setArg(1, histogramBuffer);
                    histogramKernel->setArg(2, bits);
                    histogramKernel->setArg(3, localSize, nullptr); // allocate local histogram

                    queue->enqueueKernel(histogramKernel, 1, globalWorkSizes, localWorkSizes);
                    queue->finish();

                    // Scan the histogram
                    scan_r(workGroupSize, histogramBuffer, histogramSize, true);
                    queue->finish();

                    // Permute the element to appropriate place
                    permuteKernel->setArg(0, srcBuffer);
                    permuteKernel->setArg(1, histogramBuffer);
                    permuteKernel->setArg(2, bits);
                    permuteKernel->setArg(3, localSize, nullptr);
                    permuteKernel->setArg(4, dstBuffer);

                    queue->enqueueKernel(permuteKernel, 1, globalWorkSizes, localWorkSizes);
                    queue->finish();

                    std::swap(srcBuffer, dstBuffer);
                }
            }

            /**
            * Recursive vector scan
            */
            void scan_r(size_t workGroupSize, Buffer* values, size_t size, bool first)
            {
                size_t sumBufferSize = roundToMultiple(size / (workGroupSize * 2), workGroupSize * 2);

                Buffer* sums = context->createBuffer(CL_MEM_READ_WRITE, sumBufferSize * sizeof(cl_uint));

                scanKernel->setArg(0, values);
                scanKernel->setArg(1, sums);
                scanKernel->setArg(2, sizeof(cl_uint) * 2 * workGroupSize, nullptr);
                scanKernel->setArg(3, (cl_int)first);

                size_t globalWorkSizes[] = { size / 2 }; // each thread processed 2 elements
                size_t localWorkSizes[] = { workGroupSize };

                queue->enqueueKernel(scanKernel, 1, globalWorkSizes, localWorkSizes);

                if(size > workGroupSize * 2)
                {
                    // the buffer containes more than one scanned block, scan the created sum buffer
                    scan_r(workGroupSize, sums, sumBufferSize, false);

                    // apply the sums to the buffer
                    addKernel->setArg(0, values);
                    addKernel->setArg(1, sums);
                    addKernel->setArg(2, (cl_int)first);

                    queue->enqueueKernel(addKernel, 1, globalWorkSizes, localWorkSizes);
                }

                delete sums;
            }

            void download(T* result, size_t size) override
            {
                queue->enqueueRead(srcBuffer, result, 0, size * sizeof(T));

                delete srcBuffer;
                delete histogramBuffer;
                delete dstBuffer;
            }

            void cleanup() override
            {
                delete histogramKernel;
                delete permuteKernel;
                delete scanKernel;
                delete addKernel;
            }

            virtual ~RadixSort() {}

        private:
            size_t numGroups;
            size_t bufferSize;
            size_t histogramSize;

            Kernel* histogramKernel;
            Kernel* permuteKernel;
            Kernel* scanKernel;
            Kernel* addKernel;

            Buffer* srcBuffer;
            Buffer* histogramBuffer;
            Buffer* dstBuffer;
        };
    }
}
