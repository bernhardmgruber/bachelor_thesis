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

void scan(int* data, int* result, size_t size)
{
    result[0] = data[0];
    for(size_t i = 1; i < size; i++)
        result[i] = result[i - 1] + data[i];
}

void scanCL(int* data, int* result, cl_uint size, cl_context context, cl_command_queue queue, cl_kernel kernel, size_t workGroupSize)
{
    cl_int error;

    cl_mem source      = clCreateBuffer(context, CL_MEM_READ_WRITE, size * sizeof(int), nullptr, &error);
    cl_mem destination = clCreateBuffer(context, CL_MEM_READ_WRITE, size * sizeof(int), nullptr, &error);

    error = clEnqueueWriteBuffer(queue, source, false, 0, size * sizeof(int), data, 0, nullptr, nullptr);

    size_t adjustedWorkSize = roundToMultiple(size, workGroupSize);

    for(cl_uint power = 1; power < size; power <<= 1)
    {
        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &source);
        error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &destination);
        error = clSetKernelArg(kernel, 3, sizeof(cl_uint), &power);
        error = clSetKernelArg(kernel, 3, sizeof(cl_uint), &size);

        size_t globalWorkSizes[] = { adjustedWorkSize };
        size_t localWorkSizes[] = { workGroupSize };

        error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);

        swap(source, destination);
    }

    error = clEnqueueReadBuffer(queue, source, true, 0, size * sizeof(int), result, 0, nullptr, nullptr);

    error = clReleaseMemObject(source);
    error = clReleaseMemObject(destination);
}

void scanCLWorkEfficient(int* data, int* result, cl_uint size, cl_context context, cl_command_queue queue, cl_kernel upSweep, cl_kernel setLastZero, cl_kernel downSweep, size_t workGroupSize)
{
    cl_int error;

    size_t bufferSize = roundToPowerOfTwo(size);

    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize * sizeof(int), nullptr, &error);

    error = clEnqueueWriteBuffer(queue, buffer, false, 0, size * sizeof(int), data, 0, nullptr, nullptr);

    size_t globalWorkSizes[] = { bufferSize };
    size_t localWorkSizes[] = { min(workGroupSize, bufferSize) };

    // upsweep (reduce)
    for(cl_uint offset = 1; offset < bufferSize; offset <<= 1)
    {
        cl_uint stride = 2 * offset;

        error = clSetKernelArg(upSweep, 0, sizeof(cl_mem), &buffer);
        error = clSetKernelArg(upSweep, 1, sizeof(cl_uint), &offset);
        error = clSetKernelArg(upSweep, 2, sizeof(cl_uint), &stride);

        error = clEnqueueNDRangeKernel(queue, upSweep, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);
    }

    // set last element to zero
    cl_uint lastIndex = bufferSize - 1;

    error = clSetKernelArg(setLastZero, 0, sizeof(cl_mem), &buffer);
    error = clSetKernelArg(setLastZero, 1, sizeof(cl_uint), &lastIndex);

    error = clEnqueueTask(queue, setLastZero, 0, nullptr, nullptr);

    // downsweep
    for(cl_uint offset = bufferSize >> 1; offset >= 1; offset >>= 1)
    {
        cl_uint stride = 2 * offset;

        error = clSetKernelArg(downSweep, 0, sizeof(cl_mem), &buffer);
        error = clSetKernelArg(downSweep, 1, sizeof(cl_uint), &offset);
        error = clSetKernelArg(downSweep, 2, sizeof(cl_uint), &stride);

        error = clEnqueueNDRangeKernel(queue, downSweep, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);
    }

    error = clEnqueueReadBuffer(queue, buffer, true, 0, size * sizeof(int), result, 0, nullptr, nullptr);

    error = clReleaseMemObject(buffer);
}

void scanCLRecursive_r(cl_mem values, cl_uint size, cl_context context, cl_command_queue queue, cl_kernel scanBlocks, cl_kernel addSums, size_t workGroupSize)
{
    cl_int error;

    size_t sumBufferSize = roundToMultiple(size / (workGroupSize * 2), workGroupSize * 2);

    cl_mem sums = clCreateBuffer(context, CL_MEM_READ_WRITE, sumBufferSize * sizeof(int), nullptr, &error);

    size_t globalWorkSizes[] = { size / 2 }; // the global work size is the half number of elements (each thread processed 2 elements)
    size_t localWorkSizes[] = { min(workGroupSize, globalWorkSizes[0]) };

    error = clSetKernelArg(scanBlocks, 0, sizeof(cl_mem), &values);
    error = clSetKernelArg(scanBlocks, 1, sizeof(cl_mem), &sums);
    error = clSetKernelArg(scanBlocks, 2, sizeof(int) * 2 * localWorkSizes[0], nullptr);

    error = clEnqueueNDRangeKernel(queue, scanBlocks, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);

    if(size > localWorkSizes[0] * 2)
    {
        // the buffer containes more than one scanned block, scan the created sum buffer
        scanCLRecursive_r(sums, sumBufferSize, context, queue, scanBlocks, addSums, workGroupSize);

        // apply the sums to the buffer
        error = clSetKernelArg(addSums, 0, sizeof(cl_mem), &values);
        error = clSetKernelArg(addSums, 1, sizeof(cl_mem), &sums);

        error = clEnqueueNDRangeKernel(queue, addSums, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);
    }

    clReleaseMemObject(sums);
}

void scanCLRecursive(int* data, int* result, cl_uint size, cl_context context, cl_command_queue queue, cl_kernel scanBlocks, cl_kernel addSums, size_t workGroupSize)
{
    cl_int error;

    size_t bufferSize = roundToMultiple(size, workGroupSize * 2);

    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize * sizeof(int), nullptr, &error);

    error = clEnqueueWriteBuffer(queue, buffer, false, 0, size * sizeof(int), data, 0, nullptr, nullptr);

    scanCLRecursive_r(buffer, bufferSize, context, queue, scanBlocks, addSums, workGroupSize);

    error = clEnqueueReadBuffer(queue, buffer, true, 0, size * sizeof(int), result, 0, nullptr, nullptr);

    error = clReleaseMemObject(buffer);
}

#define VECTOR_WIDTH 8

void scanCLRecursiveVector_r(cl_mem values, cl_uint size, cl_context context, cl_command_queue queue, cl_kernel scanBlocks, cl_kernel addSums, size_t workGroupSize)
{
    cl_int error;

    size_t sumBufferSize = roundToMultiple(size / (workGroupSize * 2 * VECTOR_WIDTH), workGroupSize * 2 * VECTOR_WIDTH);

    cl_mem sums = clCreateBuffer(context, CL_MEM_READ_WRITE, sumBufferSize * sizeof(int), nullptr, &error);

    size_t globalWorkSizes[] = { size / (2 * VECTOR_WIDTH) };
    size_t localWorkSizes[] = { min(workGroupSize, globalWorkSizes[0]) };

    error = clSetKernelArg(scanBlocks, 0, sizeof(cl_mem), &values);
    error = clSetKernelArg(scanBlocks, 1, sizeof(cl_mem), &sums);
    error = clSetKernelArg(scanBlocks, 2, sizeof(int) * 2 * localWorkSizes[0], nullptr);

    error = clEnqueueNDRangeKernel(queue, scanBlocks, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);

    if(size > localWorkSizes[0] * 2 * VECTOR_WIDTH)
    {
        // the buffer containes more than one scanned block, scan the created sum buffer
        scanCLRecursiveVector_r(sums, sumBufferSize, context, queue, scanBlocks, addSums, workGroupSize);

        // apply the sums to the buffer
        error = clSetKernelArg(addSums, 0, sizeof(cl_mem), &values);
        error = clSetKernelArg(addSums, 1, sizeof(cl_mem), &sums);

        error = clEnqueueNDRangeKernel(queue, addSums, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);
    }

    clReleaseMemObject(sums);
}

void scanCLRecursiveVector(int* data, int* result, cl_uint size, cl_context context, cl_command_queue queue, cl_kernel scanBlocks, cl_kernel addSums, size_t workGroupSize)
{
    cl_int error;

    size_t bufferSize = roundToMultiple(size, workGroupSize * 2 * VECTOR_WIDTH);

    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize * sizeof(int), nullptr, &error);

    error = clEnqueueWriteBuffer(queue, buffer, false, 0, size * sizeof(int), data, 0, nullptr, nullptr);

    scanCLRecursiveVector_r(buffer, bufferSize, context, queue, scanBlocks, addSums, workGroupSize);

    error = clEnqueueReadBuffer(queue, buffer, true, 0, size * sizeof(int), result, 0, nullptr, nullptr);

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
    cl_kernel kernel1 = clCreateKernel(program1, "Scan", &error);
    cl_kernel kernel2_up = clCreateKernel(program2, "UpSweep", &error);
    cl_kernel kernel2_zero = clCreateKernel(program2, "SetLastZero", &error);
    cl_kernel kernel2_down = clCreateKernel(program2, "DownSweep", &error);
    cl_kernel kernel3_scan = clCreateKernel(program3, "WorkEfficientScan", &error);
    cl_kernel kernel3_sums = clCreateKernel(program3, "AddSums", &error);
    cl_kernel kernel4_scan = clCreateKernel(program4, "WorkEfficientScan", &error);
    cl_kernel kernel4_sums = clCreateKernel(program4, "AddSums", &error);

    // SCAN
    size_t size = 1<<20;
    int* input = new int[size];
    int* result0 = new int[size]();
    int* result1 = new int[size]();
    int* result2 = new int[size]();
    int* result3 = new int[size]();
    int* result4 = new int[size]();

    generate(input, input + size, []()
    {
        return rand() % 100;
    });

    scan(input, result0, size);

    scanCL(input, result1, size, context, queue, kernel1, 256);
    if(!memcmp(result0, result1, size))
        cerr << "validation of kernel 1 failed" << endl;
    else
        cout << "kernel 1 ok" << endl;

    scanCLWorkEfficient(input, result2, size, context, queue, kernel2_up, kernel2_zero, kernel2_down, 256);
    if(!memcmp(result0, result2, size))
        cerr << "validation of kernel 2 failed" << endl;
    else
        cout << "kernel 2 ok" << endl;

    scanCLRecursive(input, result3, size, context, queue, kernel3_scan, kernel3_sums, 256);
    if(!memcmp(result0, result3, size))
        cerr << "validation of kernel 3 failed" << endl;
    else
        cout << "kernel 3 ok" << endl;

    scanCLRecursiveVector(input, result4, size, context, queue, kernel4_scan, kernel4_sums, 256);
    if(!memcmp(result0, result4, size))
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
    error = clReleaseKernel(kernel2_zero);
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
