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

#pragma once

#include "../../../common/CLAlgorithm.h"
#include "../../SortAlgorithm.h"

using namespace std;

namespace gpu
{
    namespace amd
    {
        /**
         * From: http://developer.amd.com/tools/hc/AMDAPPSDK/samples/Pages/default.aspx
         */
        template<typename T>
        class RadixSort : public CLAlgorithm<T>, public SortAlgorithm
        {
            static const unsigned int RADIX = 4;
            static const unsigned int BUCKETS = (1 << RADIX);

            public:
                const string getName() override
                {
                    return "Radix sort (AMD)";
                }

                bool isInPlace() override
                {
                    return false;
                }

                void init() override
                {
                    Program* program = context->createProgram("gpu/amd/RadixSort.cl", "-D T=" + getTypeName<T>());
                    histogramKernel = program->createKernel("histogram");
                    permuteKernel = program->createKernel("permute");
                    delete program;
                }

                void upload(size_t workGroupSize, T* data, size_t size) override
                {
                    //element count must be multiple of workGroupSize * BUCKETS
                    size_t mulFactor = workGroupSize * BUCKETS;

                    if(size < mulFactor)
                        adaptedSize = mulFactor;
                    else
                        adaptedSize = (size / mulFactor) * mulFactor;

                    numGroups = adaptedSize / mulFactor;

                    dSortedData = new T[adaptedSize]();

                    // each workgroup has it's own histogram
                    histogramSize = numGroups * workGroupSize * BUCKETS * sizeof(cl_uint);
                    histogramBins = new cl_uint[histogramSize]();

                    // Output for histogram kernel
                    unsortedDataBuf = context->createBuffer(CL_MEM_READ_ONLY, adaptedSize * sizeof(T));
                    histogramBinsBuf = context->createBuffer(CL_MEM_WRITE_ONLY, histogramSize);

                    // Input for permute kernel
                    scanedHistogramBinsBuf = context->createBuffer(CL_MEM_READ_ONLY, histogramSize);

                    // Final output
                    sortedDataBuf = context->createBuffer(CL_MEM_WRITE_ONLY, adaptedSize * sizeof(T));

                    // Allocate and init memory used by host
                    unsortedData = new T[adaptedSize];
                    memcpy(unsortedData, data, size * sizeof(T));
                }

                void run(size_t workGroupSize, size_t size) override
                {
                    for(size_t bits = 0; bits < sizeof(T) * RADIX; bits += RADIX)
                    {
                        // Calculate thread-histograms
                        runHistogramKernel(queue, bits, workGroupSize);

                        /*cout << "before" << endl;
                        for(int i = 0; i < numGroups; i++)
                            for(int j = 0; j < workGroupSize; j++)
                                printArr(histogramBins + i * workGroupSize * BUCKETS + j * BUCKETS, BUCKETS);*/

                        // Scan the histogram
                        int sum = 0;
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
                        }

                        /*cout << "after" << endl;
                        for(int i = 0; i < numGroups; i++)
                            for(int j = 0; j < workGroupSize; j++)
                                printArr(histogramBins + i * workGroupSize * BUCKETS + j * BUCKETS, BUCKETS);*/

                        // Permute the element to appropriate place
                        runPermuteKernel(queue, bits, workGroupSize);

                        // Current output now becomes the next input
                        memcpy(unsortedData, dSortedData, adaptedSize * sizeof(cl_uint));
                    }
                }

                void runHistogramKernel(CommandQueue* queue, int bits, size_t workGroupSize)
                {
                    queue->enqueueWrite(unsortedDataBuf, unsortedData);

                    size_t localSize = (workGroupSize * BUCKETS * sizeof(cl_ushort));

                    histogramKernel->setArg(0, unsortedDataBuf);
                    histogramKernel->setArg(1, histogramBinsBuf);
                    histogramKernel->setArg(2, bits);
                    histogramKernel->setArg(3, localSize, nullptr); // allocate local histogram

                    size_t globalThreads[] = { adaptedSize / BUCKETS };
                    size_t localThreads[] = { workGroupSize };

                    queue->enqueueKernel(histogramKernel, 1, globalThreads, localThreads);
                    queue->enqueueBarrier();

                    queue->enqueueRead(histogramBinsBuf, histogramBins);
                }

                void runPermuteKernel(CommandQueue* queue, int bits, size_t workGroupSize)
                {
                    queue->enqueueWrite(scanedHistogramBinsBuf, histogramBins);
                    queue->enqueueBarrier();

                    permuteKernel->setArg(0, unsortedDataBuf);
                    permuteKernel->setArg(1, scanedHistogramBinsBuf);
                    permuteKernel->setArg(2, bits);
                    permuteKernel->setArg(3, (workGroupSize * BUCKETS * sizeof(cl_ushort)), nullptr);
                    permuteKernel->setArg(4, sortedDataBuf);

                    size_t globalThreads[] = { adaptedSize / BUCKETS };
                    size_t localThreads[] = { workGroupSize };
                    queue->enqueueKernel(permuteKernel, 1, globalThreads, localThreads);

                    queue->enqueueRead(sortedDataBuf, dSortedData);
                }

                void download(T* result, size_t size) override
                {
                    memcpy(result, dSortedData, size * sizeof(T));

                    delete[] unsortedData;
                    delete[] dSortedData;
                    delete[] histogramBins;

                    delete unsortedDataBuf;
                    delete histogramBinsBuf;
                    delete scanedHistogramBinsBuf;
                    delete sortedDataBuf;
                }

                void cleanup() override
                {
                    delete histogramKernel;
                    delete permuteKernel;
                }

                virtual ~RadixSort() {}

            private:
                size_t numGroups;
                size_t adaptedSize;
                size_t histogramSize;

                T* unsortedData;
                T* dSortedData;
                cl_uint* histogramBins;

                Kernel* histogramKernel;
                Kernel* permuteKernel;

                Buffer* unsortedDataBuf;
                Buffer* histogramBinsBuf;
                Buffer* scanedHistogramBinsBuf;
                Buffer* sortedDataBuf;
        };
    }
}
