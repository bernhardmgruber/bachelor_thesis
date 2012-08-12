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

#ifndef AMDRADIXSORT_H
#define AMDRADIXSORT_H

#define RADIX 4
#define RADICES (1 << RADIX)

#include "../../GPUSortingAlgorithm.h"
//#include "../../OpenCL.h"

using namespace std;

namespace amd
{
    /**
     * From:
     */
    template<typename T, size_t count>
    class RadixSort : public GPUSortingAlgorithm<T, count>
    {
            using Base = GPUSortingAlgorithm<T, count>;

        public:
            RadixSort(Context* context, CommandQueue* queue)
                : GPUSortingAlgorithm<T, count>("Radix sort (AMD)", context, queue)
            {
            }

            virtual ~RadixSort()
            {
            }

        protected:
            bool init()
            {
                //elementCount = sampleCommon->roundToPowerOf2<cl_uint>(elementCount);
                elementCount = count;

                groupSize = Base::context->getInfoSize(CL_DEVICE_MAX_WORK_GROUP_SIZE);

                //element count must be multiple of GROUP_SIZE * RADICES
                size_t mulFactor = groupSize * RADICES;

                if(elementCount < mulFactor)
                    elementCount = mulFactor;
                else
                    elementCount = (elementCount / mulFactor) * mulFactor;

                numGroups = elementCount / mulFactor;

                // Allocate and init memory used by host
                unsortedData = SortingAlgorithm<T, count>::data;

                dSortedData = new T[elementCount]();

                hSortedData = new T[elementCount]();

                size_t tempSize = numGroups * groupSize * RADICES * sizeof(T);
                histogramBins = new T[tempSize]();

                program = Base::context->createProgram("gpu/amd/RadixSort.cl");

                histogramKernel = program->createKernel("histogram");
                permuteKernel = program->createKernel("permute");

                // Output for histogram kernel
                unsortedDataBuf = Base::context->createBuffer(CL_MEM_READ_ONLY, elementCount * sizeof(T));
                histogramBinsBuf = Base::context->createBuffer(CL_MEM_WRITE_ONLY, tempSize);

                // Input for permute kernel
                scanedHistogramBinsBuf = Base::context->createBuffer(CL_MEM_READ_ONLY, tempSize);

                // Final output
                sortedDataBuf = Base::context->createBuffer(CL_MEM_WRITE_ONLY, elementCount * sizeof(cl_uint));

                return true;
            }

            void upload()
            {
                //bfKey = Base::context->createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(T) * count, SortingAlgorithm<T, count>::data);

                //Base::queue->finish();
            }

            void sort(size_t workGroupSize)
            {
                for(size_t bits = 0; bits < sizeof(T) * RADIX; bits += RADIX)
                {
                    // Calculate thread-histograms
                    runHistogramKernel(bits, groupSize);

                    // Scan the histogram
                    int sum = 0;
                    for(size_t i = 0; i < RADICES; ++i)
                    {
                        for(size_t j = 0; j < numGroups; ++j)
                        {
                            for(size_t k = 0; k < groupSize; ++k)
                            {
                                int index = j * groupSize * RADICES + k * RADICES + i;
                                int value = histogramBins[index];
                                histogramBins[index] = sum;
                                sum += value;
                            }
                        }
                    }

                    // Permute the element to appropriate place
                    runPermuteKernel(bits, groupSize);

                    // Current output now becomes the next input
                    memcpy(unsortedData, dSortedData, elementCount * sizeof(cl_uint));
                }
            }

            void runHistogramKernel(int bits, size_t groupSize)
            {
                //cl_int status;
                //cl_int eventStatus = CL_QUEUED;
                //cl_event ndrEvt;

                /*if(localThreads > deviceInfo.maxWorkItemSizes[0] ||
                        localThreads > deviceInfo.maxWorkGroupSize)
                {
                    std::cout << "Unsupported: Device does not"
                              "support requested number of work items.";
                    return SDK_FAILURE;
                }*/

                // Enqueue write from unSortedData to unSortedDataBuf
                /*cl_event writeEvt;
                clEnqueueWriteBuffer(commandQueue, unsortedDataBuf, CL_FALSE, 0, sizeof(cl_uint) * elementCount, unsortedData, 0, NULL, &writeEvt);
                clFlush(commandQueue);
                clWaitForEvents(1, &writeEvt);
                clReleaseEvent(writeEvt);*/

                Base::queue->enqueueWrite(unsortedDataBuf, unsortedData);


                // Setup kernel arguments
                /*status = clSetKernelArg(histogramKernel,
                                        0,
                                        sizeof(cl_mem),
                                        (void *)&unsortedDataBuf);

                status = clSetKernelArg(histogramKernel,
                                        1,
                                        sizeof(cl_mem),
                                        (void *)&histogramBinsBuf);

                status = clSetKernelArg(histogramKernel,
                                        2,
                                        sizeof(cl_int),
                                        (void *)&bits);

                status = clSetKernelArg(histogramKernel,
                                        3,
                                        (groupSize * RADICES * sizeof(cl_ushort)),
                                        NULL);*/
                size_t localSize = (groupSize * RADICES * sizeof(cl_ushort));

                histogramKernel->setArg(0, unsortedDataBuf);
                histogramKernel->setArg(1, histogramBinsBuf);
                histogramKernel->setArg(2, bits);
                histogramKernel->setArg(3, localSize, nullptr);

                /*
                if(kernelInfoHistogram.localMemoryUsed > deviceInfo.localMemSize)
                {
                    std::cout << "Unsupported: Insufficient"
                              "local memory on device." << std::endl;
                    return SDK_FAILURE;
                }*/

                /*
                * Enqueue a kernel run call.
                */
                size_t globalThreads[] = { elementCount / RADICES };
                size_t localThreads[] = { groupSize };

                Base::queue->enqueueKernel(histogramKernel, 1, globalThreads, localThreads);
                /*status = clEnqueueNDRangeKernel(
                              commandQueue,
                              histogramKernel,
                              1,
                              NULL,
                              &globalThreads,
                              &localThreads,
                              0,
                              NULL,
                              &ndrEvt);*/

                //status = clFlush(commandQueue);

                //status = sampleCommon->waitForEventAndRelease(&ndrEvt);
                Base::queue->enqueueBarrier();

                // Enqueue the results to application pointer
                /*cl_event readEvt;
                status = clEnqueueReadBuffer(
                             commandQueue,
                             histogramBinsBuf,
                             CL_FALSE,
                             0,
                             numGroups * groupSize * RADICES * sizeof(cl_uint),
                             histogramBins,
                             0,
                             NULL,
                             &readEvt);*/
                Base::queue->enqueueRead(histogramBinsBuf, histogramBins);

                //status = clFlush(commandQueue);

                //status = sampleCommon->waitForEventAndRelease(&readEvt);
                Base::queue->enqueueBarrier();
            }

            void runPermuteKernel(int bits, size_t groupSize)
            {
                //cl_int status;
                //cl_int eventStatus = CL_QUEUED;
                //cl_event ndrEvt;

                //size_t bufferSize = numGroups * groupSize * RADICES * sizeof(cl_uint);

                // Write the host updated data to histogramBinsBuf
                /*cl_event writeEvt;
                status = clEnqueueWriteBuffer(commandQueue,
                                              scanedHistogramBinsBuf,
                                              CL_FALSE,
                                              0,
                                              bufferSize,
                                              histogramBins,
                                              0,
                                              NULL,
                                              &writeEvt);*/
                Base::queue->enqueueWrite(scanedHistogramBinsBuf, histogramBins);

                //status = clFlush(commandQueue);
                //status = sampleCommon->waitForEventAndRelease(&writeEvt);
                Base::queue->enqueueBarrier();

                /*if(localThreads > deviceInfo.maxWorkItemSizes[0] ||
                        localThreads > deviceInfo.maxWorkGroupSize)
                {
                    std::cout<<"Unsupported: Device does not"
                             "support requested number of work items.";
                    return SDK_FAILURE;
                }*/

                // Whether sort is to be in increasing order. CL_TRUE implies increasing
                /*status = clSetKernelArg(permuteKernel,
                                        0,
                                        sizeof(cl_mem),
                                        (void *)&unsortedDataBuf);

                status = clSetKernelArg(permuteKernel,
                                        1,
                                        sizeof(cl_mem),
                                        (void *)&scanedHistogramBinsBuf);

                status = clSetKernelArg(permuteKernel,
                                        2,
                                        sizeof(cl_int),
                                        (void *)&bits);

                status = clSetKernelArg(permuteKernel,
                                        3,
                                        (groupSize * RADICES * sizeof(cl_ushort)),
                                        NULL);

                status = clSetKernelArg(permuteKernel,
                                        4,
                                        sizeof(cl_mem),
                                        (void *)&sortedDataBuf);*/
                permuteKernel->setArg(0, unsortedDataBuf);
                permuteKernel->setArg(1, scanedHistogramBinsBuf);
                permuteKernel->setArg(2, bits);
                permuteKernel->setArg(3, (groupSize * RADICES * sizeof(cl_ushort)), nullptr);
                permuteKernel->setArg(4, sortedDataBuf);

                /*if(kernelInfoPermute.localMemoryUsed > deviceInfo.localMemSize)
                {
                    std::cout << "Unsupported: Insufficient"
                              "local memory on device." << std::endl;
                    return SDK_FAILURE;
                }*/

                /*
                * Enqueue a kernel run call.
                */
                /*status = clEnqueueNDRangeKernel(
                             commandQueue,
                             permuteKernel,
                             1,
                             NULL,
                             &globalThreads,
                             &localThreads,
                             0,
                             NULL,
                             &ndrEvt);*/

                size_t globalThreads[] = { elementCount / RADICES };
                size_t localThreads[] = { groupSize };
                Base::queue->enqueueKernel(permuteKernel, 1, globalThreads, localThreads);

                //status = clFlush(commandQueue);
                //status = sampleCommon->waitForEventAndRelease(&ndrEvt);

                //Base::queue->enqueueBarrier();

                // Enqueue the results to application pointe
                /*cl_event readEvt;
                status = clEnqueueReadBuffer(
                             commandQueue,
                             sortedDataBuf,
                             CL_FALSE,
                             0,
                             elementCount * sizeof(cl_uint),
                             dSortedData,
                             0,
                             NULL,
                             &readEvt);*/
                Base::queue->enqueueRead(sortedDataBuf, dSortedData);

                //status = clFlush(commandQueue);
                //status = sampleCommon->waitForEventAndRelease(&readEvt);
            }

            void download()
            {
                memcpy(SortingAlgorithm<T, count>::data, dSortedData, count * sizeof(T));
                //Base::queue->finish();
            }

            void cleanup()
            {
                //delete[] unsortedData;
                delete[] dSortedData;
                delete[] hSortedData;
                delete[] histogramBins;

                delete program;
                delete histogramKernel;
                delete permuteKernel;
                delete unsortedDataBuf;
                delete histogramBinsBuf;
                delete scanedHistogramBinsBuf;
                delete sortedDataBuf;
            }

            size_t numGroups;
            size_t groupSize;
            size_t elementCount;

            T* unsortedData;
            T* dSortedData;
            T* hSortedData;
            T* histogramBins;

            Program* program;
            Kernel* histogramKernel;
            Kernel* permuteKernel;
            Buffer* unsortedDataBuf;
            Buffer* histogramBinsBuf;
            Buffer* scanedHistogramBinsBuf;
            Buffer* sortedDataBuf;
    };
}

#endif // AMDRADIXSORT_H
