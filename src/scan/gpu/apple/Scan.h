//
// File:       scan.c
//
// Abstract:   This example shows how to perform an efficient parallel prefix sum (aka Scan)
//             using OpenCL.  Scan is a common data parallel primitive which can be used for
//             variety of different operations -- this example uses local memory for storing
//             partial sums and avoids memory bank conflicts on architectures which serialize
//             memory operations that are serviced on the same memory bank by offsetting the
//             loads and stores based on the size of the local group and the number of
//             memory banks (see appropriate macro definition).  As a result, this example
//             requires that the local group size > 1.
//
// Version:    <1.0>
//
// Disclaimer: IMPORTANT:  This Apple software is supplied to you by Apple Inc. ("Apple")
//             in consideration of your agreement to the following terms, and your use,
//             installation, modification or redistribution of this Apple software
//             constitutes acceptance of these terms.  If you do not agree with these
//             terms, please do not use, install, modify or redistribute this Apple
//             software.
//
//             In consideration of your agreement to abide by the following terms, and
//             subject to these terms, Apple grants you a personal, non - exclusive
//             license, under Apple's copyrights in this original Apple software ( the
//             "Apple Software" ), to use, reproduce, modify and redistribute the Apple
//             Software, with or without modifications, in source and / or binary forms;
//             provided that if you redistribute the Apple Software in its entirety and
//             without modifications, you must retain this notice and the following text
//             and disclaimers in all such redistributions of the Apple Software. Neither
//             the name, trademarks, service marks or logos of Apple Inc. may be used to
//             endorse or promote products derived from the Apple Software without specific
//             prior written permission from Apple.  Except as expressly stated in this
//             notice, no other rights or licenses, express or implied, are granted by
//             Apple herein, including but not limited to any patent rights that may be
//             infringed by your derivative works or by other works in which the Apple
//             Software may be incorporated.
//
//             The Apple Software is provided by Apple on an "AS IS" basis.  APPLE MAKES NO
//             WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION THE IMPLIED
//             WARRANTIES OF NON - INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A
//             PARTICULAR PURPOSE, REGARDING THE APPLE SOFTWARE OR ITS USE AND OPERATION
//             ALONE OR IN COMBINATION WITH YOUR PRODUCTS.
//
//             IN NO EVENT SHALL APPLE BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL OR
//             CONSEQUENTIAL DAMAGES ( INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//             SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//             INTERRUPTION ) ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION, MODIFICATION
//             AND / OR DISTRIBUTION OF THE APPLE SOFTWARE, HOWEVER CAUSED AND WHETHER
//             UNDER THEORY OF CONTRACT, TORT ( INCLUDING NEGLIGENCE ), STRICT LIABILITY OR
//             OTHERWISE, EVEN IF APPLE HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright ( C ) 2008 Apple Inc. All Rights Reserved.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../ScanAlgorithm.h"
#include "../../../common/GPUAlgorithm.h"

#define NUM_BANKS       (32)

namespace gpu
{
    namespace apple
    {
        enum KernelMethods
        {
            PRESCAN                             = 0,
            PRESCAN_STORE_SUM                   = 1,
            PRESCAN_STORE_SUM_NON_POWER_OF_TWO  = 2,
            PRESCAN_NON_POWER_OF_TWO            = 3,
            UNIFORM_ADD                         = 4
        };

        static const char* KernelNames[] =
        {
            "PreScanKernel",
            "PreScanStoreSumKernel",
            "PreScanStoreSumNonPowerOfTwoKernel",
            "PreScanNonPowerOfTwoKernel",
            "UniformAddKernel"
        };

        static const unsigned int KernelCount = sizeof(KernelNames) / sizeof(char *);

        template<typename T>
        class Scan : public GPUAlgorithm<T>, public ScanAlgorithm
        {
            public:
                const string getName() override
                {
                    return "Scan (apple) (exclusiv)";
                }

                bool isInclusiv() override
                {
                    return false;
                }

                bool IsPowerOfTwo(int n)
                {
                    return ((n&(n-1))==0) ;
                }

                int floorPow2(int n)
                {
                    int exp;
                    frexp((float)n, &exp);
                    return 1 << (exp - 1);
                }

                int CreatePartialSumBuffers(Context* context, unsigned int size)
                {
                    ElementsAllocated = size;

                    unsigned int group_size = GROUP_SIZE;
                    unsigned int element_count = size;

                    int level = 0;

                    do
                    {
                        unsigned int group_count = (int)fmax(1, (int)ceil((float)element_count / (2.0f * group_size)));
                        if (group_count > 1)
                        {
                            level++;
                        }
                        element_count = group_count;
                    }
                    while (element_count > 1);

                    ScanPartialSums = new Buffer*[level];
                    LevelsAllocated = level;
                    memset(ScanPartialSums, 0, sizeof(Buffer*) * level);

                    element_count = size;
                    level = 0;

                    do
                    {
                        unsigned int group_count = (int)fmax(1, (int)ceil((float)element_count / (2.0f * group_size)));
                        if (group_count > 1)
                        {
                            size_t buffer_size = group_count * sizeof(float);
                            ScanPartialSums[level++] = context->createBuffer(CL_MEM_READ_WRITE, buffer_size);
                        }

                        element_count = group_count;

                    }
                    while (element_count > 1);

                    return CL_SUCCESS;
                }

                void ReleasePartialSums()
                {
                    unsigned int i;
                    for (i = 0; i < LevelsAllocated; i++)
                    {
                        delete ScanPartialSums[i];
                    }

                    delete ScanPartialSums;
                    ScanPartialSums = 0;
                    ElementsAllocated = 0;
                    LevelsAllocated = 0;
                }

                int PreScan(CommandQueue* queue, size_t *global, size_t *local, size_t shared, Buffer* output_data, Buffer* input_data, unsigned int n, int group_index, int base_index)
                {
                    unsigned int k = PRESCAN;
                    unsigned int a = 0;
                    kernels[k]->setArg(a++, output_data);
                    kernels[k]->setArg(a++, input_data);
                    kernels[k]->setArg(a++, shared, nullptr);
                    kernels[k]->setArg(a++, (cl_int) group_index);
                    kernels[k]->setArg(a++, (cl_int) base_index);
                    kernels[k]->setArg(a++, (cl_int) n);

                    queue->enqueueKernel(kernels[k], 1, global, local);

                    return CL_SUCCESS;
                }

                int PreScanStoreSum(CommandQueue* queue, size_t *global, size_t *local, size_t shared, Buffer* output_data, Buffer* input_data, Buffer* partial_sums, unsigned int n, int group_index, int base_index)
                {
                    unsigned int k = PRESCAN_STORE_SUM;
                    unsigned int a = 0;
                    kernels[k]->setArg(a++, output_data);
                    kernels[k]->setArg(a++, input_data);
                    kernels[k]->setArg(a++, partial_sums);
                    kernels[k]->setArg(a++, shared, nullptr);
                    kernels[k]->setArg(a++, (cl_int) group_index);
                    kernels[k]->setArg(a++, (cl_int) base_index);
                    kernels[k]->setArg(a++, (cl_int) n);
                    queue->enqueueKernel(kernels[k], 1, global, local);

                    return CL_SUCCESS;
                }

                int PreScanStoreSumNonPowerOfTwo(CommandQueue* queue, size_t *global, size_t *local, size_t shared, Buffer* output_data, Buffer* input_data, Buffer* partial_sums, unsigned int n, int group_index, int base_index)
                {
                    unsigned int k = PRESCAN_STORE_SUM_NON_POWER_OF_TWO;
                    unsigned int a = 0;
                    kernels[k]->setArg(a++, output_data);
                    kernels[k]->setArg(a++, input_data);
                    kernels[k]->setArg(a++, partial_sums);
                    kernels[k]->setArg(a++, shared, nullptr);
                    kernels[k]->setArg(a++, (cl_int)group_index);
                    kernels[k]->setArg(a++, (cl_int)base_index);
                    kernels[k]->setArg(a++, (cl_int)n);
                    queue->enqueueKernel(kernels[k], 1, global, local);

                    return CL_SUCCESS;
                }

                int PreScanNonPowerOfTwo(CommandQueue* queue, size_t *global, size_t *local, size_t shared, Buffer* output_data, Buffer* input_data, unsigned int n, int group_index, int base_index)
                {
                    unsigned int k = PRESCAN_NON_POWER_OF_TWO;
                    unsigned int a = 0;
                    kernels[k]->setArg(a++, output_data);
                    kernels[k]->setArg(a++, input_data);
                    kernels[k]->setArg(a++, shared, nullptr);
                    kernels[k]->setArg(a++, (cl_int)group_index);
                    kernels[k]->setArg(a++, (cl_int)base_index);
                    kernels[k]->setArg(a++, (cl_int)n);
                    queue->enqueueKernel(kernels[k], 1, global, local);

                    return CL_SUCCESS;
                }

                int UniformAdd(CommandQueue* queue, size_t *global, size_t *local, Buffer* output_data, Buffer* partial_sums, unsigned int n, unsigned int group_offset, unsigned int base_index)
                {
#if DEBUG_INFO
                    printf("UniformAdd: Global[%4d] Local[%4d] BlockOffset[%4d] BaseIndex[%4d] Entries[%d]\n",
                           (int)global[0], (int)local[0], group_offset, base_index, n);
#endif

                    unsigned int k = UNIFORM_ADD;
                    unsigned int a = 0;

                    /*int err = CL_SUCCESS;
                    err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_mem), &output_data);
                    err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_mem), &partial_sums);
                    err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(float),  0);
                    err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_int), &group_offset);
                    err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_int), &base_index);
                    err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_int), &n);
                    if (err != CL_SUCCESS)
                    {
                        printf("Error: %s: Failed to set kernel arguments!\n", KernelNames[k]);
                        return EXIT_FAILURE;
                    }*/
                    kernels[k]->setArg(a++, output_data);
                    kernels[k]->setArg(a++, partial_sums);
                    kernels[k]->setArg(a++, sizeof(T), nullptr);
                    kernels[k]->setArg(a++, (cl_int)group_offset);
                    kernels[k]->setArg(a++, (cl_int)base_index);
                    kernels[k]->setArg(a++, (cl_int)n);

                    /*err = CL_SUCCESS;
                    err |= clEnqueueNDRangeKernel(ComputeCommands, ComputeKernels[k], 1, NULL, global, local, 0, NULL, NULL);
                    if (err != CL_SUCCESS)
                    {
                        printf("Error: %s: Failed to execute kernel!\n", KernelNames[k]);
                        return EXIT_FAILURE;
                    }*/
                    queue->enqueueKernel(kernels[k], 1, global, local);

                    return CL_SUCCESS;
                }

                int PreScanBufferRecursive(CommandQueue* queue, Buffer* output_data, Buffer* input_data, unsigned int max_group_size, unsigned int max_work_item_count, unsigned int element_count, int level)
                {
                    unsigned int group_size = max_group_size;
                    unsigned int group_count = (int)fmax(1.0f, (int)ceil((float)element_count / (2.0f * group_size)));
                    unsigned int work_item_count = 0;

                    if (group_count > 1)
                        work_item_count = group_size;
                    else if (IsPowerOfTwo(element_count))
                        work_item_count = element_count / 2;
                    else
                        work_item_count = floorPow2(element_count);

                    work_item_count = (work_item_count > max_work_item_count) ? max_work_item_count : work_item_count;

                    unsigned int element_count_per_group = work_item_count * 2;
                    unsigned int last_group_element_count = element_count - (group_count-1) * element_count_per_group;
                    unsigned int remaining_work_item_count = (int)fmax(1.0f, last_group_element_count / 2);
                    remaining_work_item_count = (remaining_work_item_count > max_work_item_count) ? max_work_item_count : remaining_work_item_count;
                    unsigned int remainder = 0;
                    size_t last_shared = 0;


                    if (last_group_element_count != element_count_per_group)
                    {
                        remainder = 1;

                        if(!IsPowerOfTwo(last_group_element_count))
                            remaining_work_item_count = floorPow2(last_group_element_count);

                        remaining_work_item_count = (remaining_work_item_count > max_work_item_count) ? max_work_item_count : remaining_work_item_count;
                        unsigned int padding = (2 * remaining_work_item_count) / NUM_BANKS;
                        last_shared = sizeof(float) * (2 * remaining_work_item_count + padding);
                    }

                    remaining_work_item_count = (remaining_work_item_count > max_work_item_count) ? max_work_item_count : remaining_work_item_count;
                    size_t global[] = { (int)fmax(1, group_count - remainder) * work_item_count, 1 };
                    size_t local[]  = { work_item_count, 1 };

                    unsigned int padding = element_count_per_group / NUM_BANKS;
                    size_t shared = sizeof(float) * (element_count_per_group + padding);

                    Buffer* partial_sums = ScanPartialSums[level];
                    int err = CL_SUCCESS;

                    if (group_count > 1)
                    {
                        err = PreScanStoreSum(queue, global, local, shared, output_data, input_data, partial_sums, work_item_count * 2, 0, 0);
                        if(err != CL_SUCCESS)
                            return err;

                        if (remainder)
                        {
                            size_t last_global[] = { 1 * remaining_work_item_count, 1 };
                            size_t last_local[]  = { remaining_work_item_count, 1 };

                            err = PreScanStoreSumNonPowerOfTwo(queue,
                                                               last_global, last_local, last_shared,
                                                               output_data, input_data, partial_sums,
                                                               last_group_element_count,
                                                               group_count - 1,
                                                               element_count - last_group_element_count);

                            if(err != CL_SUCCESS)
                                return err;

                        }

                        err = PreScanBufferRecursive(queue, partial_sums, partial_sums, max_group_size, max_work_item_count, group_count, level + 1);
                        if(err != CL_SUCCESS)
                            return err;

                        err = UniformAdd(queue, global, local, output_data, partial_sums,  element_count - last_group_element_count, 0, 0);
                        if(err != CL_SUCCESS)
                            return err;

                        if (remainder)
                        {
                            size_t last_global[] = { 1 * remaining_work_item_count, 1 };
                            size_t last_local[]  = { remaining_work_item_count, 1 };

                            err = UniformAdd(queue,
                                             last_global, last_local,
                                             output_data, partial_sums,
                                             last_group_element_count,
                                             group_count - 1,
                                             element_count - last_group_element_count);

                            if(err != CL_SUCCESS)
                                return err;
                        }
                    }
                    else if (IsPowerOfTwo(element_count))
                    {
                        err = PreScan(queue, global, local, shared, output_data, input_data, work_item_count * 2, 0, 0);
                        if(err != CL_SUCCESS)
                            return err;
                    }
                    else
                    {
                        err = PreScanNonPowerOfTwo(queue, global, local, shared, output_data, input_data, element_count, 0, 0);
                        if(err != CL_SUCCESS)
                            return err;
                    }

                    return CL_SUCCESS;
                }

                void PreScanBuffer(CommandQueue* queue, Buffer* output_data, Buffer* input_data, unsigned int max_group_size, unsigned int max_work_item_count, unsigned int element_count)
                {
                    PreScanBufferRecursive(queue, output_data, input_data, max_group_size, max_work_item_count, element_count, 0);
                }
                void init(Context* context) override
                {
                    size_t max_workgroup_size = context->getInfo<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE);

                    GROUP_SIZE = min( GROUP_SIZE, max_workgroup_size );

                    Program* program = context->createProgram("gpu/apple/Scan.cl", "-D T=" + getTypeName<T>());

                    //ComputeKernels = (cl_kernel*) malloc(KernelCount * sizeof(cl_kernel));
                    kernels = new Kernel*[KernelCount];
                    for(size_t i = 0; i < KernelCount; i++)
                    {
                        // Create each compute kernel from within the program
                        kernels[i] = program->createKernel(KernelNames[i]);
                        size_t wgSize = kernels[i]->getWorkGroupSize();
                        GROUP_SIZE = min( GROUP_SIZE, wgSize );
                    }

                    delete program;
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
                {
                    // Create the input buffer on the device
                    size_t buffer_size = sizeof(T) * size;

                    input = context->createBuffer(CL_MEM_READ_WRITE, buffer_size);
                    queue->enqueueWrite(input, data);

                    T* zeroMem = new T[size];
                    memset(zeroMem, 0, sizeof(T) * size);
                    output = context->createBuffer(CL_MEM_READ_WRITE, buffer_size);
                    queue->enqueueWrite(output, zeroMem);
                    delete[] zeroMem;
                }

                void run(CommandQueue* queue, size_t workGroupSize, size_t size) override
                {
                    CreatePartialSumBuffers(queue->getContext(), size);
                    PreScanBuffer(queue, output, input, GROUP_SIZE, GROUP_SIZE, size);
                }

                void download(CommandQueue* queue, T* result, size_t size) override
                {
                    queue->enqueueRead(output, result);
                    delete input;
                    delete output;
                }

                void cleanup() override
                {
                    ReleasePartialSums();
                    for(size_t  i = 0; i < KernelCount; i++)
                        delete kernels[i];
                    delete[] kernels;
                }

                virtual ~Scan() {}

            private:
                Kernel** kernels;
                Buffer* input;
                Buffer* output;

                size_t GROUP_SIZE = 256;
                Buffer** ScanPartialSums = 0;
                unsigned int ElementsAllocated = 0;
                unsigned int LevelsAllocated = 0;
        };
    }
}
