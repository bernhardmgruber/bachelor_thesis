#include <CL/cl.h>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <exception>
#include <stdexcept>
#include <stdint.h>

#define int cl_int

using namespace std;

int roundToMultiple(int x, int multiple)
{
    if(x % multiple == 0)
        return x;
    else
        return (x / multiple + 1) * multiple;
}

cl_uint roundToPowerOfTwo(cl_uint x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;

    return x + 1;
}

typedef uint32_t uint;

void sortC(uint* data, size_t n)
{
    qsort(data, n, sizeof(uint), [] (const void* a, const void* b) -> int
    {
        if (*((uint*)a) < *((uint*)b))
            return -1;
        else if (*((uint*)a) > *((uint*)b))
            return 1;
        return 0;
    });
}

void sortCPP(uint* data, size_t n)
{
    std::sort(data, data + n);
}

#define RADIX 16
#define BUCKETS (1 << RADIX)
#define RADIX_MASK (BUCKETS - 1)

void radixSort(uint* data, size_t n)
{
    uint* aux = new uint[n];

    size_t histogram[BUCKETS];

    uint* src = data;
    uint* dst = aux;

    for(size_t bits = 0; bits < sizeof(uint) * 8 ; bits += RADIX)
    {
        memset(histogram, 0, BUCKETS * sizeof(size_t));

        // calculate histogram
        for(size_t i = 0; i < n; ++i)
        {
            uint element = src[i];
            uint pos = (element >> bits) & RADIX_MASK;
            histogram[pos]++;
        }

        // scan histogram (exclusive)
        size_t sum = 0;
        for(size_t i = 0; i < BUCKETS; ++i)
        {
            size_t val = histogram[i];
            histogram[i] = sum;
            sum += val;
        }

        // permute
        for(size_t i = 0; i < n; ++i)
        {
            uint element = src[i];
            uint pos = (element >> bits) & RADIX_MASK;

            size_t index = histogram[pos]++;
            dst[index] = src[i];
        }

        std::swap(src, dst);
    }

    if(dst != data)
        memcpy(data, dst, n * sizeof(uint));

    delete[] aux;
}

void bitonicSort(uint* data, cl_uint n, cl_context context, cl_command_queue queue, cl_kernel kernel, size_t workGroupSize)
{
    cl_int error;

    size_t bufferSize = roundToPowerOfTwo(n);
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize * sizeof(int), nullptr, &error);

    error = clEnqueueWriteBuffer(queue, buffer, false, 0, n * sizeof(uint), data, 0, nullptr, nullptr);

    if(bufferSize != n)
    {
        cl_uint max = numeric_limits<uint>::max();
        error = clEnqueueFillBuffer(queue, buffer, &max, sizeof(uint), n * sizeof(uint), (bufferSize - n) * sizeof(uint), 0, nullptr, nullptr);
    }

    for (cl_uint boxwidth = 2; boxwidth <= bufferSize; boxwidth <<= 1)
    {
        for (cl_uint inc = boxwidth >> 1; inc > 0; inc >>= 1)
        {
            error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
            error = clSetKernelArg(kernel, 1, sizeof(cl_uint), &inc);
            error = clSetKernelArg(kernel, 2, sizeof(cl_uint), &boxwidth);

            size_t nThreads = bufferSize / 2;
            size_t globalWorkSizes[] = { nThreads };
            size_t localWorkSizes[] = { min(workGroupSize, nThreads) };

            error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);
        }
    }

    clEnqueueReadBuffer(queue, buffer, true, 0, n * sizeof(uint), data, 0, nullptr, nullptr);

    clReleaseMemObject(buffer);
}

void bitonicSortFusion(uint* data, cl_uint n, cl_context context, cl_command_queue queue, cl_kernel kernel2, cl_kernel kernel4, cl_kernel kernel8, cl_kernel kernel16, size_t workGroupSize)
{
    cl_int error;

    size_t bufferSize = roundToPowerOfTwo(n);
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize * sizeof(uint), nullptr, &error);

    error = clEnqueueWriteBuffer(queue, buffer, false, 0, n * sizeof(uint), data, 0, nullptr, nullptr);

    if(bufferSize != n)
    {
        cl_uint max = numeric_limits<uint>::max();
        error = clEnqueueFillBuffer(queue, buffer, &max, sizeof(uint), n * sizeof(uint), (bufferSize - n) * sizeof(uint), 0, nullptr, nullptr);
    }

    for (cl_uint boxwidth = 2; boxwidth <= bufferSize; boxwidth <<= 1)
    {
        for (cl_uint inc = boxwidth >> 1; inc > 0; )
        {
            int ninc = 0;
            cl_kernel kernel;

            if (inc >= 8)
            {
                kernel = kernel16;
                ninc = 4;
            }
            else if (inc >= 4)
            {
                kernel = kernel8;
                ninc = 3;
            }
            else if (inc >= 2)
            {
                kernel = kernel4;
                ninc = 2;
            }
            else
            {
                kernel = kernel2;
                ninc = 1;
            }

            error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
            error = clSetKernelArg(kernel, 1, sizeof(cl_uint), &inc);
            error = clSetKernelArg(kernel, 2, sizeof(cl_uint), &boxwidth);

            size_t nThreads = bufferSize >> ninc;
            size_t globalWorkSizes[] = { nThreads };
            size_t localWorkSizes[] = { min(workGroupSize, nThreads) };

            error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);

            inc >>= ninc;
        }
    }

    clEnqueueReadBuffer(queue, buffer, true, 0, n * sizeof(uint), data, 0, nullptr, nullptr);

    clReleaseMemObject(buffer);
}

#define VECTOR_WIDTH 8

void scanCLRecursiveVector_r(cl_mem values, cl_uint n, cl_context context, cl_command_queue queue, cl_kernel scanBlocksKernel, cl_kernel addSumsKernel, size_t workGroupSize)
{
    cl_int error;

    size_t sumBufferSize = roundToMultiple(n / (workGroupSize * 2 * VECTOR_WIDTH), workGroupSize * 2 * VECTOR_WIDTH);

    cl_mem sums = clCreateBuffer(context, CL_MEM_READ_WRITE, sumBufferSize * sizeof(int), nullptr, &error);

    error = clSetKernelArg(scanBlocksKernel, 0, sizeof(cl_mem), &values);
    error = clSetKernelArg(scanBlocksKernel, 1, sizeof(cl_mem), &sums);
    error = clSetKernelArg(scanBlocksKernel, 2, sizeof(int) * 2 * workGroupSize, nullptr);

    size_t globalWorkSizes[] = { n / (2 * VECTOR_WIDTH) };
    size_t localWorkSizes[] = { workGroupSize };

    error = clEnqueueNDRangeKernel(queue, scanBlocksKernel, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);

    if(n > workGroupSize * 2 * VECTOR_WIDTH)
    {
        scanCLRecursiveVector_r(sums, sumBufferSize, context, queue, scanBlocksKernel, addSumsKernel, workGroupSize);

        error = clSetKernelArg(addSumsKernel, 0, sizeof(cl_mem), &values);
        error = clSetKernelArg(addSumsKernel, 1, sizeof(cl_mem), &sums);

        error = clEnqueueNDRangeKernel(queue, addSumsKernel, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);
    }

    clReleaseMemObject(sums);
}

#undef RADIX
#undef BUCKETS
#undef RADIX_MASK
#undef VECTOR_WIDTH

#define RADIX 4
#define BUCKETS (1 << RADIX)
#define RADIX_MASK (BUCKETS - 1)
#define BLOCK_SIZE 32
#define VECTOR_WIDTH 8

void radixSortCL(uint* data, cl_uint n, cl_context context, cl_command_queue queue, cl_kernel histogramKernel, cl_kernel permuteKernel, cl_kernel scanBlocksKernel, cl_kernel addSumsKernel, size_t workGroupSize)
{
    cl_int error;

    size_t bufferSize = roundToMultiple(n, workGroupSize * BLOCK_SIZE);

    cl_mem srcBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize * sizeof(uint), nullptr, &error);
    cl_mem dstBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize * sizeof(uint), nullptr, &error);

    error = clEnqueueWriteBuffer(queue, srcBuffer, false, 0, n * sizeof(uint), data, 0, nullptr, nullptr);

    if(bufferSize != n)
    {
        cl_uint max = numeric_limits<uint>::max();
        error = clEnqueueFillBuffer(queue, srcBuffer, &max, sizeof(uint), n * sizeof(uint), (bufferSize - n) * sizeof(uint), 0, nullptr, nullptr);
    }

    size_t histogramSize = (bufferSize / BLOCK_SIZE) * BUCKETS;
    histogramSize = roundToMultiple(histogramSize, workGroupSize * 2 * VECTOR_WIDTH); // for scan

    cl_mem histogramBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, histogramSize * sizeof(cl_uint), nullptr, &error);

    size_t globalWorkSizes[] = { bufferSize / BLOCK_SIZE };
    size_t localWorkSizes[] = { workGroupSize };

    for(cl_uint bits = 0; bits < sizeof(uint) * 8; bits += RADIX)
    {
        error = clSetKernelArg(histogramKernel, 0, sizeof(cl_mem), &srcBuffer);
        error = clSetKernelArg(histogramKernel, 1, sizeof(cl_mem), &histogramBuffer);
        error = clSetKernelArg(histogramKernel, 2, sizeof(cl_uint), &bits);

        error = clEnqueueNDRangeKernel(queue, histogramKernel, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);

        scanCLRecursiveVector_r(histogramBuffer, histogramSize, context, queue, scanBlocksKernel, addSumsKernel, workGroupSize);

        error = clSetKernelArg(permuteKernel, 0, sizeof(cl_mem), &srcBuffer);
        error = clSetKernelArg(permuteKernel, 1, sizeof(cl_mem), &dstBuffer);
        error = clSetKernelArg(permuteKernel, 2, sizeof(cl_mem), &histogramBuffer);
        error = clSetKernelArg(permuteKernel, 3, sizeof(cl_uint), &bits);

        error = clEnqueueNDRangeKernel(queue, permuteKernel, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);

        std::swap(srcBuffer, dstBuffer);
    }

    clEnqueueReadBuffer(queue, srcBuffer, true, 0, n * sizeof(uint), data, 0, nullptr, nullptr);

    clReleaseMemObject(srcBuffer);
    clReleaseMemObject(dstBuffer);
    clReleaseMemObject(histogramBuffer);
}

void radixSortCLLocal(uint* data, cl_uint n, cl_context context, cl_command_queue queue, cl_kernel histogramKernel, cl_kernel permuteKernel, cl_kernel scanBlocksKernel, cl_kernel addSumsKernel, size_t workGroupSize)
{
    cl_int error;

    size_t bufferSize = roundToMultiple(n, workGroupSize * BLOCK_SIZE);

    cl_mem srcBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize * sizeof(uint), nullptr, &error);
    cl_mem dstBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize * sizeof(uint), nullptr, &error);

    error = clEnqueueWriteBuffer(queue, srcBuffer, false, 0, n * sizeof(uint), data, 0, nullptr, nullptr);

    if(bufferSize != n)
    {
        cl_uint max = numeric_limits<uint>::max();
        error = clEnqueueFillBuffer(queue, srcBuffer, &max, sizeof(uint), n * sizeof(uint), (bufferSize - n) * sizeof(uint), 0, nullptr, nullptr);
    }

    // each thread has it's own histogram
    size_t histogramSize = (bufferSize / BLOCK_SIZE) * BUCKETS;
    histogramSize = roundToMultiple(histogramSize, workGroupSize * 2 * VECTOR_WIDTH); // for scan

    cl_mem histogramBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, histogramSize * sizeof(cl_uint), nullptr, &error);

    size_t localSize = (workGroupSize * BUCKETS * sizeof(cl_uint));

    size_t globalWorkSizes[] = { bufferSize / BLOCK_SIZE };
    size_t localWorkSizes[] = { workGroupSize };

    for(cl_uint bits = 0; bits < sizeof(uint) * 8; bits += RADIX)
    {
        // Calculate thread-histograms
        error = clSetKernelArg(histogramKernel, 0, sizeof(cl_mem), &srcBuffer);
        error = clSetKernelArg(histogramKernel, 1, sizeof(cl_mem), &histogramBuffer);
        error = clSetKernelArg(histogramKernel, 2, sizeof(cl_uint), &bits);
        error = clSetKernelArg(histogramKernel, 3, localSize, nullptr);

        error = clEnqueueNDRangeKernel(queue, histogramKernel, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);

        // Scan the histogram
        scanCLRecursiveVector_r(histogramBuffer, histogramSize, context, queue, scanBlocksKernel, addSumsKernel, workGroupSize);

        // Permute the element to appropriate place
        error = clSetKernelArg(permuteKernel, 0, sizeof(cl_mem), &srcBuffer);
        error = clSetKernelArg(permuteKernel, 1, sizeof(cl_mem), &dstBuffer);
        error = clSetKernelArg(permuteKernel, 2, sizeof(cl_mem), &histogramBuffer);
        error = clSetKernelArg(permuteKernel, 3, sizeof(cl_uint), &bits);
        error = clSetKernelArg(permuteKernel, 4, localSize, nullptr);

        error = clEnqueueNDRangeKernel(queue, permuteKernel, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);

        std::swap(srcBuffer, dstBuffer);
    }

    clEnqueueReadBuffer(queue, srcBuffer, true, 0, n * sizeof(uint), data, 0, nullptr, nullptr);

    clReleaseMemObject(srcBuffer);
    clReleaseMemObject(dstBuffer);
    clReleaseMemObject(histogramBuffer);
}

#if 0
    size_t localSize = (workGroupSize * BUCKETS * sizeof(cl_uint));
    ...
    for(cl_uint bits = 0; bits < sizeof(uint) * 8; bits += RADIX)
        ...
        error = clSetKernelArg(histogramKernel, 3, localSize, nullptr);
        ...
        error = clSetKernelArg(permuteKernel, 4, localSize, nullptr);
#endif

string readFile(string fileName)
{
    ifstream file(fileName, ios::in);
    if(!file)
        throw runtime_error("Error opening file " + fileName);

    string buffer = string((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());

    file.close();

    return buffer;
}

int main(int argc, char* argv[])
{
    cl_int error;

    // get the first available platform
    cl_platform_id platform;
    error = clGetPlatformIDs(1, &platform, nullptr);

    // get the first available GPU on this platform
    cl_device_id device;
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    // create context
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);

    // create command queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error);

    // create OpenCL program from source code
    string source1 = readFile("../../../src/sort/gpu/thesis/BitonicSort.cl");
    string source2 = readFile("../../../src/sort/gpu/thesis/BitonicSortFusion.cl");
    string source3 = readFile("../../../src/sort/gpu/thesis/RadixSort.cl");
    string source4 = readFile("../../../src/sort/gpu/thesis/RadixSortLocal.cl");

    const char* source1Ptr = source1.c_str();
    const char* source2Ptr = source2.c_str();
    const char* source3Ptr = source3.c_str();
    const char* source4Ptr = source4.c_str();

    cl_program program1 = clCreateProgramWithSource(context, 1, &source1Ptr, nullptr, &error);
    cl_program program2 = clCreateProgramWithSource(context, 1, &source2Ptr, nullptr, &error);
    cl_program program3 = clCreateProgramWithSource(context, 1, &source3Ptr, nullptr, &error);
    cl_program program4 = clCreateProgramWithSource(context, 1, &source4Ptr, nullptr, &error);

    // compile the program for the device
    error = clBuildProgram(program1, 1, &device, "", nullptr, nullptr);
    error = clBuildProgram(program2, 1, &device, "", nullptr, nullptr);
    error = clBuildProgram(program3, 1, &device, "", nullptr, nullptr);
    error = clBuildProgram(program4, 1, &device, "", nullptr, nullptr);

    // create the kernel
    cl_kernel kernel1 = clCreateKernel(program1, "BitonicSort", &error);
    cl_kernel kernel2_2 = clCreateKernel(program2, "BitonicSortFusion2", &error);
    cl_kernel kernel2_4 = clCreateKernel(program2, "BitonicSortFusion4", &error);
    cl_kernel kernel2_8 = clCreateKernel(program2, "BitonicSortFusion8", &error);
    cl_kernel kernel2_16 = clCreateKernel(program2, "BitonicSortFusion16", &error);
    cl_kernel kernel3_hist = clCreateKernel(program3, "Histogram", &error);
    cl_kernel kernel3_perm = clCreateKernel(program3, "Permute", &error);
    cl_kernel kernel3_scan = clCreateKernel(program3, "ScanBlocksVec", &error);
    cl_kernel kernel3_sums = clCreateKernel(program3, "AddSums", &error);
    cl_kernel kernel4_hist = clCreateKernel(program4, "Histogram", &error);
    cl_kernel kernel4_perm = clCreateKernel(program4, "Permute", &error);
    cl_kernel kernel4_scan = clCreateKernel(program4, "ScanBlocksVec", &error);
    cl_kernel kernel4_sums = clCreateKernel(program4, "AddSums", &error);

    // SCAN
    size_t n = 1<<20;
    uint* input = new uint[n];
    uint* buffer0 = new uint[n]();
    uint* buffer1 = new uint[n]();
    uint* buffer2 = new uint[n]();
    uint* buffer3 = new uint[n]();
    uint* buffer4 = new uint[n]();
    uint* buffer5 = new uint[n]();
    uint* buffer6 = new uint[n]();

    generate(input, input + n, []()
    {
        return rand() % 100;
    });

    copy(input, input + n, buffer0);
    copy(input, input + n, buffer1);
    copy(input, input + n, buffer2);
    copy(input, input + n, buffer3);
    copy(input, input + n, buffer4);
    copy(input, input + n, buffer5);
    copy(input, input + n, buffer6);

    sortCPP(buffer0, n);

    sortC(buffer1, n);
    if(memcmp(buffer0, buffer1, n * sizeof(uint)))
        cerr << "validation of qsort() failed" << endl;
    else
        cout << "qsort() ok" << endl;

    radixSort(buffer2, n);
    if(memcmp(buffer0, buffer2, n * sizeof(uint)))
        cerr << "validation of radixSort() failed" << endl;
    else
        cout << "radixSort() ok" << endl;

    bitonicSort(buffer3, n, context, queue, kernel1, 256);
    if(memcmp(buffer0, buffer3, n * sizeof(uint)))
        cerr << "validation of bitonic sort failed" << endl;
    else
        cout << "bitonic sort ok" << endl;

    bitonicSortFusion(buffer4, n, context, queue, kernel2_2, kernel2_4, kernel2_8, kernel2_16, 256);
    if(memcmp(buffer0, buffer4, n * sizeof(uint)))
        cerr << "validation of bitonic sort fusion failed" << endl;
    else
        cout << "bitonic sort fusion ok" << endl;

    radixSortCL(buffer5, n, context, queue, kernel3_hist, kernel3_perm, kernel3_scan, kernel3_sums, 256);
    if(memcmp(buffer0, buffer5, n * sizeof(uint)))
        cerr << "validation of radix sort failed" << endl;
    else
        cout << "radix sort ok" << endl;

    radixSortCLLocal(buffer6, n, context, queue, kernel4_hist, kernel4_perm, kernel4_scan, kernel4_sums, 256);
    if(memcmp(buffer0, buffer6, n * sizeof(uint)))
        cerr << "validation of radix sort local failed" << endl;
    else
        cout << "radix sort local ok" << endl;

    delete[] input;
    delete[] buffer0;
    delete[] buffer1;
    delete[] buffer2;
    delete[] buffer3;
    delete[] buffer4;
    delete[] buffer5;
    delete[] buffer6;

    error = clReleaseKernel(kernel1);
    error = clReleaseKernel(kernel2_2);
    error = clReleaseKernel(kernel2_4);
    error = clReleaseKernel(kernel2_8);
    error = clReleaseKernel(kernel2_16);
    error = clReleaseKernel(kernel3_hist);
    error = clReleaseKernel(kernel3_perm);
    error = clReleaseKernel(kernel3_scan);
    error = clReleaseKernel(kernel3_sums);
    error = clReleaseKernel(kernel4_hist);
    error = clReleaseKernel(kernel4_perm);
    error = clReleaseKernel(kernel4_scan);
    error = clReleaseKernel(kernel4_sums);

    error = clReleaseProgram(program1);
    error = clReleaseProgram(program2);
    error = clReleaseProgram(program3);
    error = clReleaseProgram(program4);

    error = clReleaseCommandQueue(queue);
    error = clReleaseContext(context);
    error = clReleaseDevice(device);

    return 0;
}
