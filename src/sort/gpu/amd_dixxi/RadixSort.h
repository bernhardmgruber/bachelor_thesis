/* ============================================================

Copyright (c) 2009 Advanced Micro Devices, Inc.  All rights reserved.

Redistribution and use of this material is permitted under the following
conditions:

Redistributions must retain the above copyright notice and all terms of this
license.

In no event shall anyone redistributing or accessing or using this material
commence or participate in any arbitration or legal action relating to this
material against Advanced Micro Devices, Inc. or any copyright holders or
contributors. The foregoing shall survive any expiration or termination of
this license or any agreement or access or use related to this material.

ANY BREACH OF ANY TERM OF THIS LICENSE SHALL RESULT IN THE IMMEDIATE REVOCATION
OF ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE THIS MATERIAL.

THIS MATERIAL IS PROVIDED BY ADVANCED MICRO DEVICES, INC. AND ANY COPYRIGHT
HOLDERS AND CONTRIBUTORS "AS IS" IN ITS CURRENT CONDITION AND WITHOUT ANY
REPRESENTATIONS, GUARANTEE, OR WARRANTY OF ANY KIND OR IN ANY WAY RELATED TO
SUPPORT, INDEMNITY, ERROR FREE OR UNINTERRUPTED OPERA TION, OR THAT IT IS FREE
FROM DEFECTS OR VIRUSES.  ALL OBLIGATIONS ARE HEREBY DISCLAIMED - WHETHER
EXPRESS, IMPLIED, OR STATUTORY - INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED
WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
ACCURACY, COMPLETENESS, OPERABILITY, QUALITY OF SERVICE, OR NON-INFRINGEMENT.
IN NO EVENT SHALL ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, PUNITIVE,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, REVENUE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED OR BASED ON ANY THEORY OF LIABILITY
ARISING IN ANY WAY RELATED TO THIS MATERIAL, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE. THE ENTIRE AND AGGREGATE LIABILITY OF ADVANCED MICRO DEVICES,
INC. AND ANY COPYRIGHT HOLDERS AND CONTRIBUTORS SHALL NOT EXCEED TEN DOLLARS
(US $10.00). ANYONE REDISTRIBUTING OR ACCESSING OR USING THIS MATERIAL ACCEPTS
THIS ALLOCATION OF RISK AND AGREES TO RELEASE ADVANCED MICRO DEVICES, INC. AND
ANY COPYRIGHT HOLDERS AND CONTRIBUTORS FROM ANY AND ALL LIABILITIES,
OBLIGATIONS, CLAIMS, OR DEMANDS IN EXCESS OF TEN DOLLARS (US $10.00). THE
FOREGOING ARE ESSENTIAL TERMS OF THIS LICENSE AND, IF ANY OF THESE TERMS ARE
CONSTRUED AS UNENFORCEABLE, FAIL IN ESSENTIAL PURPOSE, OR BECOME VOID OR
DETRIMENTAL TO ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR
CONTRIBUTORS FOR ANY REASON, THEN ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE
THIS MATERIAL SHALL TERMINATE IMMEDIATELY. MOREOVER, THE FOREGOING SHALL
SURVIVE ANY EXPIRATION OR TERMINATION OF THIS LICENSE OR ANY AGREEMENT OR
ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE IS HEREBY PROVIDED, AND BY REDISTRIBUTING OR ACCESSING OR USING THIS
MATERIAL SUCH NOTICE IS ACKNOWLEDGED, THAT THIS MATERIAL MAY BE SUBJECT TO
RESTRICTIONS UNDER THE LAWS AND REGULATIONS OF THE UNITED STATES OR OTHER
COUNTRIES, WHICH INCLUDE BUT ARE NOT LIMITED TO, U.S. EXPORT CONTROL LAWS SUCH
AS THE EXPORT ADMINISTRATION REGULATIONS AND NATIONAL SECURITY CONTROLS AS
DEFINED THEREUNDER, AS WELL AS STATE DEPARTMENT CONTROLS UNDER THE U.S.
MUNITIONS LIST. THIS MATERIAL MAY NOT BE USED, RELEASED, TRANSFERRED, IMPORTED,
EXPORTED AND/OR RE-EXPORTED IN ANY MANNER PROHIBITED UNDER ANY APPLICABLE LAWS,
INCLUDING U.S. EXPORT CONTROL LAWS REGARDING SPECIFICALLY DESIGNATED PERSONS,
COUNTRIES AND NATIONALS OF COUNTRIES SUBJECT TO NATIONAL SECURITY CONTROLS.
MOREOVER, THE FOREGOING SHALL SURVIVE ANY EXPIRATION OR TERMINATION OF ANY
LICENSE OR AGREEMENT OR ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE REGARDING THE U.S. GOVERNMENT AND DOD AGENCIES: This material is
provided with "RESTRICTED RIGHTS" and/or "LIMITED RIGHTS" as applicable to
computer software and technical data, respectively. Use, duplication,
distribution or disclosure by the U.S. Government and/or DOD agencies is
subject to the full extent of restrictions in all applicable regulations,
including those found at FAR52.227 and DFARS252.227 et seq. and any successor
regulations thereof. Use of this material by the U.S. Government and/or DOD
agencies is acknowledgment of the proprietary rights of any copyright holders
and contributors, including those of Advanced Micro Devices, Inc., as well as
the provisions of FAR52.227-14 through 23 regarding privately developed and/or
commercial computer software.

This license forms the entire agreement regarding the subject matter hereof and
supersedes all proposals and prior discussions and writings between the parties
with respect thereto. This license does not affect any ownership, rights, title,
or interest in, or relating to, this material. No terms of this license can be
modified or waived, and no breach of this license can be excused, unless done
so in a writing signed by all affected parties. Each term of this license is
separately enforceable. If any term of this license is determined to be or
becomes unenforceable or illegal, such term shall be reformed to the minimum
extent necessary in order for this license to remain in effect in accordance
with its terms as modified by such reformation. This license shall be governed
by and construed in accordance with the laws of the State of Texas without
regard to rules on conflicts of law of any state or jurisdiction or the United
Nations Convention on the International Sale of Goods. All disputes arising out
of this license shall be subject to the jurisdiction of the federal and state
courts in Austin, Texas, and all defenses are hereby waived concerning personal
jurisdiction and venue of these courts.

============================================================ */

#ifndef AMDDIXXIRADIXSORT_H
#define AMDDIXXIRADIXSORT_H

#include "../../../common/GPUAlgorithm.h"
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
        class RadixSort : public GPUAlgorithm<T>, public SortAlgorithm
        {
            static const unsigned int RADIX = 4;
            static const unsigned int BUCKETS = (1 << RADIX);

            public:
                const string getName() override
                {
                    return "Radix sort (AMD/dixxi)";
                }

                bool isInPlace() override
                {
                    return false;
                }

                void init(Context* context) override
                {
                    Program* program = context->createProgram("gpu/amd_dixxi/RadixSort.cl", "-D T=" + getTypeName<T>());
                    histogramKernel = program->createKernel("histogram");
                    permuteKernel = program->createKernel("permute");
                    delete program;

                    program = context->createProgram("gpu/amd_dixxi/LocalScan.cl", "-D T=uint");
                    scanKernel = program->createKernel("LocalScan");
                    addKernel = program->createKernel("AddSums");
                    delete program;
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
                {
                    //element count must be multiple of workGroupSize * BUCKETS
                    size_t multiple = workGroupSize * BUCKETS;

                    if(size < multiple)
                        adaptedSize = multiple;
                    else
                        adaptedSize = (size / multiple) * multiple;

                    numGroups = adaptedSize / multiple;

                    // each thread has it's own histogram
                    histogramSize = numGroups * workGroupSize * BUCKETS * sizeof(cl_uint);

                    unsortedDataBuf = context->createBuffer(CL_MEM_READ_ONLY, adaptedSize * sizeof(T));
                    queue->enqueueWrite(unsortedDataBuf, data, 0, size * sizeof(T));
                    if(adaptedSize > size)
                    {
                        T* zero = new T[adaptedSize - size];
                        memset(zero, 0, (adaptedSize - size) * sizeof(T));
                        queue->enqueueWrite(unsortedDataBuf, zero, size * sizeof(T), (adaptedSize - size) * sizeof(T));
                        delete zero;
                    }

                    histogramBinsBuf = context->createBuffer(CL_MEM_READ_WRITE, histogramSize);
                    sortedDataBuf = context->createBuffer(CL_MEM_WRITE_ONLY, adaptedSize * sizeof(T));
                }

                void run(CommandQueue* queue, size_t workGroupSize, size_t size) override
                {
                    for(size_t bits = 0; bits < sizeof(T) * RADIX; bits += RADIX)
                    {
                        // Calculate thread-histograms
                        runHistogramKernel(queue, bits, workGroupSize);

                        // Scan the histogram
                        /*int sum = 0;
                        for(size_t b = 0; b < BUCKETS; ++b)
                        {
                            for(size_t g = 0; g < numGroups; ++g)
                            {
                                for(size_t i = 0; i < workGroupSize; ++i)
                                {
                                    int index = g * workGroupSize * BUCKETS + i * BUCKETS + b;
                                    int value = histogramBins[index];
                                    histogramBins[index] = sum;
                                    sum += value;
                                }
                            }
                        }*/

                        scan_r(queue->getContext(), queue, workGroupSize, histogramBinsBuf, true);
                        queue->enqueueBarrier();

                        // Permute the element to appropriate place
                        runPermuteKernel(queue, bits, workGroupSize);

                        queue->enqueueCopy(sortedDataBuf, unsortedDataBuf);
                    }
                }

                void runHistogramKernel(CommandQueue* queue, int bits, size_t workGroupSize)
                {
                    size_t localSize = (workGroupSize * BUCKETS * sizeof(cl_ushort));

                    histogramKernel->setArg(0, unsortedDataBuf);
                    histogramKernel->setArg(1, histogramBinsBuf);
                    histogramKernel->setArg(2, bits);
                    histogramKernel->setArg(3, localSize, nullptr); // allocate local histogram

                    size_t globalThreads[] = { adaptedSize / BUCKETS };
                    size_t localThreads[] = { workGroupSize };

                    queue->enqueueKernel(histogramKernel, 1, globalThreads, localThreads);
                    queue->enqueueBarrier();
                }

                void runPermuteKernel(CommandQueue* queue, int bits, size_t workGroupSize)
                {
                    permuteKernel->setArg(0, unsortedDataBuf);
                    permuteKernel->setArg(1, histogramBinsBuf);
                    permuteKernel->setArg(2, bits);
                    permuteKernel->setArg(3, (workGroupSize * BUCKETS * sizeof(cl_ushort)), nullptr);
                    permuteKernel->setArg(4, sortedDataBuf);

                    size_t globalThreads[] = { adaptedSize / BUCKETS };
                    size_t localThreads[] = { workGroupSize };
                    queue->enqueueKernel(permuteKernel, 1, globalThreads, localThreads);
                }

                void scan_r(Context* context, CommandQueue* queue, size_t workGroupSize, Buffer* blocks, bool first)
                {
                    Buffer* sums = context->createBuffer(CL_MEM_READ_WRITE, roundToMultiple(blocks->getSize() / workGroupSize, workGroupSize * 2 * sizeof(cl_uint)));

                    size_t globalWorkSizes[] = { blocks->getSize() / sizeof(cl_uint) / 2 }; // the global work size is the half number of elements (each thread processed 2 elements)
                    size_t localWorkSizes[] = { min(workGroupSize, globalWorkSizes[0]) };

                    scanKernel->setArg(0, blocks);
                    scanKernel->setArg(1, sums);
                    scanKernel->setArg(2, sizeof(cl_uint) * 2 * localWorkSizes[0], nullptr);
                    scanKernel->setArg<cl_short>(3, first ? 1 : 0);
                    queue->enqueueKernel(scanKernel, 1, globalWorkSizes, localWorkSizes);

                    if(blocks->getSize() / sizeof(cl_uint) > localWorkSizes[0] * 2)
                    {
                        // the buffer containes more than one scanned block, scan the created sum buffer
                        scan_r(context, queue, workGroupSize, sums, false);

                        // get the remaining available local memory
                        size_t totalGlobalWorkSize = blocks->getSize() / sizeof(cl_uint) / 2;
                        size_t maxLocalMemSize = context->getInfo<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE); // FIXME: This does not work for some NVIDIA cards, raises OUT_OF_RESOURCES
                        size_t maxGlobalWorkSize = maxLocalMemSize / sizeof(cl_int) * workGroupSize;

                        size_t offset = 0;

                        do
                        {
                            globalWorkSizes[0] = min(totalGlobalWorkSize - offset, maxGlobalWorkSize);
                            localWorkSizes[0] = min(workGroupSize, globalWorkSizes[0]);
                            size_t globalWorkOffsets[] = { offset };

                            // apply the sums to the buffer
                            addKernel->setArg(0, blocks);
                            addKernel->setArg(1, sums);
                            addKernel->setArg(2, min(sums->getSize(), maxLocalMemSize), nullptr);
                            addKernel->setArg(3, (cl_short)first);
                            queue->enqueueKernel(addKernel, 1, globalWorkSizes, localWorkSizes, globalWorkOffsets);

                            offset += maxGlobalWorkSize;
                        }
                        while(offset < totalGlobalWorkSize);
                    }

                    delete sums;
                }

                void download(CommandQueue* queue, T* result, size_t size) override
                {
                    queue->enqueueRead(sortedDataBuf, result, 0, size);

                    delete unsortedDataBuf;
                    delete histogramBinsBuf;
                    delete sortedDataBuf;
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
                size_t adaptedSize;
                size_t histogramSize;

                Kernel* histogramKernel;
                Kernel* permuteKernel;
                Kernel* scanKernel;
                Kernel* addKernel;

                Buffer* unsortedDataBuf;
                Buffer* histogramBinsBuf;
                Buffer* sortedDataBuf;
        };
    }
}

#endif // AMDDIXXIRADIXSORT_H
