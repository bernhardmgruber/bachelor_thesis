#pragma once

#include "../../../common/CLAlgorithm.h"
#include "../../SortAlgorithm.h"

using namespace std;

namespace gpu
{
    namespace thesis
    {
        /**
        * From: http://developer.amd.com/tools/hc/AMDAPPSDK/samples/Pages/default.aspx
        * Modified algorithm by Bernhard Manfred Gruber.
        */
        template<typename T>
        class RadixSortLocalVec : public CLAlgorithm<T>, public SortAlgorithm
        {
            static_assert(is_same<T, cl_uint>::value, "Thesis algorithms only support 32 bit unsigned int");

            static const unsigned int RADIX = 4;
            static const unsigned int BUCKETS = (1 << RADIX);
            static const unsigned int BLOCK_SIZE = 128; // elements per thread

            static const unsigned int VECTOR_WIDTH = 8; // for recursive vector scan

        public:
            const string getName() override
            {
                return "Radix sort local vec (THESIS AMD dixxi)";
            }

            bool isInPlace() override
            {
                return false;
            }

            void init() override
            {
                Program* program = context->createProgram("gpu/thesis/RadixSortLocalVec.cl");
                histogramKernel = program->createKernel("HistogramBlock");
                permuteKernel = program->createKernel("PermuteBlock");
                scanKernel = program->createKernel("ScanBlocksVec");
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
                histogramSize = roundToMultiple(histogramSize, workGroupSize * 2 * VECTOR_WIDTH);

                histogramBuffer = context->createBuffer(CL_MEM_READ_WRITE, histogramSize * sizeof(cl_uint));
            }

            void run(size_t workGroupSize, size_t size) override
            {
                size_t localSize = (workGroupSize * BUCKETS * sizeof(cl_uint));

                size_t globalWorkSizes[] = { bufferSize / BLOCK_SIZE };
                size_t localWorkSizes[] = { workGroupSize };

                for(cl_uint bits = 0; bits < sizeof(T) * 8; bits += RADIX)
                {
                    // Calculate thread-histograms
                    histogramKernel->setArg(0, srcBuffer);
                    histogramKernel->setArg(1, histogramBuffer);
                    histogramKernel->setArg(2, bits);
                    histogramKernel->setArg(3, localSize, nullptr);

                    queue->enqueueKernel(histogramKernel, 1, globalWorkSizes, localWorkSizes);

                    // Scan the histogram
                    scan_r(workGroupSize, histogramBuffer, histogramSize);

                    // Permute the element to appropriate place
                    permuteKernel->setArg(0, srcBuffer);
                    permuteKernel->setArg(1, dstBuffer);
                    permuteKernel->setArg(2, histogramBuffer);
                    permuteKernel->setArg(3, bits);
                    permuteKernel->setArg(4, localSize, nullptr);

                    queue->enqueueKernel(permuteKernel, 1, globalWorkSizes, localWorkSizes);

                    std::swap(srcBuffer, dstBuffer);
                }
            }

            /**
            * Recursive vector scan
            */
            void scan_r(size_t workGroupSize, Buffer* values, size_t size)
            {
                size_t sumBufferSize = roundToMultiple(size / (workGroupSize * 2 * VECTOR_WIDTH), workGroupSize * 2 * VECTOR_WIDTH);

                Buffer* sums = context->createBuffer(CL_MEM_READ_WRITE, sumBufferSize * sizeof(cl_uint));

                scanKernel->setArg(0, values);
                scanKernel->setArg(1, sums);
                scanKernel->setArg(2, sizeof(cl_uint) * 2 * workGroupSize, nullptr);

                size_t globalWorkSizes[] = { size / (2 * VECTOR_WIDTH) }; // each thread processed 2 elements
                size_t localWorkSizes[] = { workGroupSize };

                queue->enqueueKernel(scanKernel, 1, globalWorkSizes, localWorkSizes);

                if(size > workGroupSize * 2 * VECTOR_WIDTH)
                {
                    // the buffer containes more than one scanned block, scan the created sum buffer
                    scan_r(workGroupSize, sums, sumBufferSize);

                    // apply the sums to the buffer
                    addKernel->setArg(0, values);
                    addKernel->setArg(1, sums);

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

            virtual ~RadixSortLocalVec() {}

        private:
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
