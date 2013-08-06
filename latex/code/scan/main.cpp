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

void scan(int* data, int* result, size_t n)
{
    result[0] = 0;
    for(size_t i = 1; i < n; i++)
        result[i] = result[i - 1] + data[i - 1];
}

void scanCL(int* data, int* result, cl_uint n, cl_context context, cl_command_queue queue, cl_kernel kernel, size_t workGroupSize)
{
    cl_int error;

    cl_mem src = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(int), nullptr, &error);
    cl_mem dst = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(int), nullptr, &error);

    error = clEnqueueWriteBuffer(queue, src, false, 0, n * sizeof(int), data, 0, nullptr, nullptr);

    size_t adjustedWorkSize = roundToMultiple(n, workGroupSize);

    for(cl_uint offset = 1; offset < n; offset <<= 1)
    {
        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
        error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst);
        error = clSetKernelArg(kernel, 2, sizeof(cl_uint), &offset);
        error = clSetKernelArg(kernel, 3, sizeof(cl_uint), &n);

        size_t globalWorkSizes[] = { adjustedWorkSize };
        size_t localWorkSizes[] = { workGroupSize };

        error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);

        swap(src, dst);
    }

    error = clEnqueueReadBuffer(queue, src, true, 0, n * sizeof(int), result, 0, nullptr, nullptr);

    error = clReleaseMemObject(src);
    error = clReleaseMemObject(dst);
}

void scanCLWorkEfficient(int* data, int* result, cl_uint n, cl_context context, cl_command_queue queue, cl_kernel upSweep, cl_kernel downSweep, size_t workGroupSize)
{
    cl_int error;

    size_t bufferSize = roundToPowerOfTwo(n);

    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize * sizeof(int), nullptr, &error);

    error = clEnqueueWriteBuffer(queue, buffer, false, 0, n * sizeof(int), data, 0, nullptr, nullptr);

    // upsweep (reduce)
    size_t nodes = bufferSize >> 1;
    for(cl_uint offset = 1; offset < bufferSize; offset <<= 1, nodes >>= 1)
    {
        error = clSetKernelArg(upSweep, 0, sizeof(cl_mem), &buffer);
        error = clSetKernelArg(upSweep, 1, sizeof(cl_uint), &offset);

        size_t globalWorkSizes[] = { nodes };
        size_t localWorkSizes[] = { min(workGroupSize, nodes) };

        error = clEnqueueNDRangeKernel(queue, upSweep, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);
    }

    // set last element to zero
    cl_uint zero = 0;
    clEnqueueWriteBuffer(queue, buffer, false, (bufferSize - 1) * sizeof(cl_uint), sizeof(cl_uint), &zero, 0, nullptr, nullptr);

    // downsweep
    nodes = 1;
    for(cl_uint offset = bufferSize >> 1; offset >= 1; offset >>= 1, nodes <<= 1)
    {
        error = clSetKernelArg(downSweep, 0, sizeof(cl_mem), &buffer);
        error = clSetKernelArg(downSweep, 1, sizeof(cl_uint), &offset);

        size_t globalWorkSizes[] = { nodes };
        size_t localWorkSizes[] = { min(workGroupSize, nodes) };

        error = clEnqueueNDRangeKernel(queue, downSweep, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);
    }

    error = clEnqueueReadBuffer(queue, buffer, true, 0, n * sizeof(int), result, 0, nullptr, nullptr);

    error = clReleaseMemObject(buffer);
}

void scanCLRecursive_r(cl_mem values, cl_uint n, cl_context context, cl_command_queue queue, cl_kernel scanBlocks, cl_kernel addSums, size_t workGroupSize)
{
    cl_int error;

    size_t sumBufferSize = roundToMultiple(n / (workGroupSize * 2), workGroupSize * 2);

    cl_mem sums = clCreateBuffer(context, CL_MEM_READ_WRITE, sumBufferSize * sizeof(int), nullptr, &error);

    error = clSetKernelArg(scanBlocks, 0, sizeof(cl_mem), &values);
    error = clSetKernelArg(scanBlocks, 1, sizeof(cl_mem), &sums);
    error = clSetKernelArg(scanBlocks, 2, sizeof(int) * 2 * workGroupSize, nullptr);

    size_t globalWorkSizes[] = { n / 2 };
    size_t localWorkSizes[] = { workGroupSize };

    error = clEnqueueNDRangeKernel(queue, scanBlocks, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);

    if(n > workGroupSize * 2)
    {
        scanCLRecursive_r(sums, sumBufferSize, context, queue, scanBlocks, addSums, workGroupSize);

        error = clSetKernelArg(addSums, 0, sizeof(cl_mem), &values);
        error = clSetKernelArg(addSums, 1, sizeof(cl_mem), &sums);

        error = clEnqueueNDRangeKernel(queue, addSums, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);
    }

    clReleaseMemObject(sums);
}

void scanCLRecursive(int* data, int* result, cl_uint n, cl_context context, cl_command_queue queue, cl_kernel scanBlocks, cl_kernel addSums, size_t workGroupSize)
{
    cl_int error;

    size_t bufferSize = roundToMultiple(n, workGroupSize * 2);

    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize * sizeof(int), nullptr, &error);

    error = clEnqueueWriteBuffer(queue, buffer, false, 0, n * sizeof(int), data, 0, nullptr, nullptr);

    scanCLRecursive_r(buffer, bufferSize, context, queue, scanBlocks, addSums, workGroupSize);

    error = clEnqueueReadBuffer(queue, buffer, true, 0, n * sizeof(int), result, 0, nullptr, nullptr);

    error = clReleaseMemObject(buffer);
}

#if 0
#define VECTOR_WIDTH 8
...
    size_t sumBufferSize = roundToMultiple(n / (workGroupSize * 2 * VECTOR_WIDTH), workGroupSize * 2 * VECTOR_WIDTH);
    ...
    size_t globalWorkSizes[] = { n / (2 * VECTOR_WIDTH) };
    ...
    if(n > workGroupSize * 2 * VECTOR_WIDTH)
    {
        scanCLRecursiveVector_r(sums, sumBufferSize, context, queue, scanBlocks, addSums, workGroupSize);
        ...
#endif // 0

#if 0
#define CONCAT(a, b) a ## b
#define CONCAT_EXPANDED(a, b) CONCAT(a, b)

#define UPSWEEP_STEP(left, right) right += left

#define UPSWEEP_STEPS(left, right) \
    UPSWEEP_STEP(CONCAT_EXPANDED(val1.s, left), CONCAT_EXPANDED(val1.s, right)); \
    UPSWEEP_STEP(CONCAT_EXPANDED(val2.s, left), CONCAT_EXPANDED(val2.s, right))

#define DOWNSWEEP_STEP_TMP(left, right, tmp) \
    int tmp = left;                          \
    left = right;                            \
    right += tmp

#define DOWNSWEEP_STEP(left, right) DOWNSWEEP_STEP_TMP(left, right, CONCAT_EXPANDED(tmp, __COUNTER__))

#define DOWNSWEEP_STEPS(left, right) \
    DOWNSWEEP_STEP(CONCAT_EXPANDED(val1.s, left), CONCAT_EXPANDED(val1.s, right)); \
    DOWNSWEEP_STEP(CONCAT_EXPANDED(val2.s, left), CONCAT_EXPANDED(val2.s, right))

__kernel void ScanBlocksVec(__global int8* buffer, __global int* sums, __local int* shared)
{
    uint globalId = get_global_id(0);
    uint localId = get_local_id(0);
    uint n = get_local_size(0) * 2;

    uint offset = 1;

    int8 val1 = buffer[2 * globalId + 0];
    int8 val2 = buffer[2 * globalId + 1];

    // upsweep vectors
    UPSWEEP_STEPS(0, 1);
    UPSWEEP_STEPS(2, 3);
    UPSWEEP_STEPS(4, 5);
    UPSWEEP_STEPS(6, 7);

    UPSWEEP_STEPS(1, 3);
    UPSWEEP_STEPS(5, 7);

    UPSWEEP_STEPS(3, 7);

    // move sums into shared memory block and clear last elements
    shared[2 * localId + 0] = val1.s7;
    shared[2 * localId + 1] = val2.s7;

    val1.s7 = 0;
    val2.s7 = 0;

    // downsweep vectors
    DOWNSWEEP_STEPS(3, 7);

    DOWNSWEEP_STEPS(1, 3);
    DOWNSWEEP_STEPS(5, 7);

    DOWNSWEEP_STEPS(0, 1);
    DOWNSWEEP_STEPS(2, 3);
    DOWNSWEEP_STEPS(4, 5);
    DOWNSWEEP_STEPS(6, 7);

    // scan block in local memory
    ...

    // apply the sums
    val1 += shared[2 * localId + 0];
    val2 += shared[2 * localId + 1];

    // write results to device memory
    buffer[2 * globalId + 0] = val1;
    buffer[2 * globalId + 1] = val2;
}

__kernel void AddSums(__global int8* buffer, __global int* sums)
{
    uint globalId = get_global_id(0);

    int val = sums[get_group_id(0)];

    buffer[globalId * 2 + 0] += val;
    buffer[globalId * 2 + 1] += val;
}
#endif // 0

#define VECTOR_WIDTH 8

void scanCLRecursiveVector_r(cl_mem values, cl_uint n, cl_context context, cl_command_queue queue, cl_kernel scanBlocks, cl_kernel addSums, size_t workGroupSize)
{
    cl_int error;

    size_t sumBufferSize = roundToMultiple(n / (workGroupSize * 2 * VECTOR_WIDTH), workGroupSize * 2 * VECTOR_WIDTH);

    cl_mem sums = clCreateBuffer(context, CL_MEM_READ_WRITE, sumBufferSize * sizeof(int), nullptr, &error);

    error = clSetKernelArg(scanBlocks, 0, sizeof(cl_mem), &values);
    error = clSetKernelArg(scanBlocks, 1, sizeof(cl_mem), &sums);
    error = clSetKernelArg(scanBlocks, 2, sizeof(int) * 2 * workGroupSize, nullptr);

    size_t globalWorkSizes[] = { n / (2 * VECTOR_WIDTH) };
    size_t localWorkSizes[] = { workGroupSize };

    error = clEnqueueNDRangeKernel(queue, scanBlocks, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);

    if(n > workGroupSize * 2 * VECTOR_WIDTH)
    {
        scanCLRecursiveVector_r(sums, sumBufferSize, context, queue, scanBlocks, addSums, workGroupSize);

        error = clSetKernelArg(addSums, 0, sizeof(cl_mem), &values);
        error = clSetKernelArg(addSums, 1, sizeof(cl_mem), &sums);

        error = clEnqueueNDRangeKernel(queue, addSums, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);
    }

    clReleaseMemObject(sums);
}

void scanCLRecursiveVector(int* data, int* result, cl_uint n, cl_context context, cl_command_queue queue, cl_kernel scanBlocks, cl_kernel addSums, size_t workGroupSize)
{
    cl_int error;

    size_t bufferSize = roundToMultiple(n, workGroupSize * 2 * VECTOR_WIDTH);

    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize * sizeof(int), nullptr, &error);

    error = clEnqueueWriteBuffer(queue, buffer, false, 0, n * sizeof(int), data, 0, nullptr, nullptr);

    scanCLRecursiveVector_r(buffer, bufferSize, context, queue, scanBlocks, addSums, workGroupSize);

    error = clEnqueueReadBuffer(queue, buffer, true, 0, n * sizeof(int), result, 0, nullptr, nullptr);

    error = clReleaseMemObject(buffer);
}

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
    string source1 = readFile("../../../src/scan/gpu/thesis/NaiveScan.cl");
    string source2 = readFile("../../../src/scan/gpu/thesis/WorkEfficientScan.cl");
    string source3 = readFile("../../../src/scan/gpu/thesis/RecursiveScan.cl");
    string source4 = readFile("../../../src/scan/gpu/thesis/RecursiveVecScan.cl");

    const char* source1Ptr = source1.c_str();
    const char* source2Ptr = source2.c_str();
    const char* source3Ptr = source3.c_str();
    const char* source4Ptr = source4.c_str();
//
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
    cl_kernel kernel1 = clCreateKernel(program1, "NaiveScan", &error);
    cl_kernel kernel2_up = clCreateKernel(program2, "UpSweep", &error);
    cl_kernel kernel2_down = clCreateKernel(program2, "DownSweep", &error);
    cl_kernel kernel3_scan = clCreateKernel(program3, "ScanBlocks", &error);
    cl_kernel kernel3_sums = clCreateKernel(program3, "AddSums", &error);
    cl_kernel kernel4_scan = clCreateKernel(program4, "ScanBlocksVec", &error);
    cl_kernel kernel4_sums = clCreateKernel(program4, "AddSums", &error);

    // SCAN
    size_t n = 1<<20;
    int* input = new int[n];
    int* result0 = new int[n]();
    int* result1 = new int[n]();
    int* result2 = new int[n]();
    int* result3 = new int[n]();
    int* result4 = new int[n]();

    generate(input, input + n, []()
    {
        return rand() % 100;
    });

    scan(input, result0, n);

    scanCL(input, result1, n, context, queue, kernel1, 256);
    if(memcmp(result0 + 1, result1, (n - 1) * sizeof(int))) // inclusive scan
        cerr << "validation of kernel 1 failed" << endl;
    else
        cout << "kernel 1 ok" << endl;

    scanCLWorkEfficient(input, result2, n, context, queue, kernel2_up, kernel2_down, 256);
    if(memcmp(result0, result2, n * sizeof(int)))
        cerr << "validation of kernel 2 failed" << endl;
    else
        cout << "kernel 2 ok" << endl;

    scanCLRecursive(input, result3, n, context, queue, kernel3_scan, kernel3_sums, 256);
    if(memcmp(result0, result3, n * sizeof(int)))
        cerr << "validation of kernel 3 failed" << endl;
    else
        cout << "kernel 3 ok" << endl;

    scanCLRecursiveVector(input, result4, n, context, queue, kernel4_scan, kernel4_sums, 256);
    if(memcmp(result0, result4, n * sizeof(int)))
        cerr << "validation of kernel 4 failed" << endl;
    else
        cout << "kernel 4 ok" << endl;

    delete[] input;
    delete[] result0;
    delete[] result1;
    delete[] result2;
    delete[] result3;
    delete[] result4;

    error = clReleaseKernel(kernel1);
    error = clReleaseKernel(kernel2_up);
    error = clReleaseKernel(kernel2_down);
    error = clReleaseKernel(kernel3_scan);
    error = clReleaseKernel(kernel3_sums);
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
