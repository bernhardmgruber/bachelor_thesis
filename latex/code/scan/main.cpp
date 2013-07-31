#include <CL/cl.h>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <exception>
#include <stdexcept>

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

    for(cl_uint power = 1; power < bufferSize; power <<= 1)
    {
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &source);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &destination);
        clSetKernelArg(kernel, 3, sizeof(cl_uint), &power);
        clSetKernelArg(kernel, 3, sizeof(cl_uint), &size);

        size_t globalWorkSizes[] = { adjustedWorkSize };
        size_t localWorkSizes[] = { workGroupSize };

        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);

        swap(source, destination);
    }

    error = clEnqueueReadBuffer(queue, source, true, 0, size * sizeof(int), result, 0, nullptr, nullptr);

    clReleaseMemObject(source);
    clReleaseMemObject(destination);
}

void scanCLWorkEfficient(int* data, int* result, cl_uint size, cl_context context, cl_command_queue queue, cl_kernel kernel, size_t workGroupSize)
{
    cl_int error;

    cl_mem source      = clCreateBuffer(context, CL_MEM_READ_WRITE, size * sizeof(int), nullptr, &error);
    cl_mem destination = clCreateBuffer(context, CL_MEM_READ_WRITE, size * sizeof(int), nullptr, &error);

    error = clEnqueueWriteBuffer(queue, source, false, 0, size * sizeof(int), data, 0, nullptr, nullptr);

    size_t adjustedWorkSize = roundToPowerOfTwo(size, workGroupSize);

    for(cl_uint power = 1; power < bufferSize; power <<= 1)
    {
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &source);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &destination);
        clSetKernelArg(kernel, 3, sizeof(cl_uint), &power);
        clSetKernelArg(kernel, 3, sizeof(cl_uint), &size);

        size_t globalWorkSizes[] = { adjustedWorkSize };
        size_t localWorkSizes[] = { workGroupSize };

        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);

        swap(source, destination);
    }

    error = clEnqueueReadBuffer(queue, source, true, 0, size * sizeof(int), result, 0, nullptr, nullptr);

    clReleaseMemObject(source);
    clReleaseMemObject(destination);
}

//
//#define BLOCK_SIZE 4
//
//void matrixMulCLRect(float* a, float* b, float* c, cl_uint size, cl_context context, cl_command_queue queue, cl_kernel kernel, size_t workGroupSize, int multiple, int elementsPerThread)
//{
//    cl_int error;
//
//    cl_uint adjustedSize = roundToMultiple(size, multiple);
//
//    size_t count = adjustedSize * adjustedSize;
//
//    cl_mem aBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, count * sizeof(float), nullptr, &error);
//    cl_mem bBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, count * sizeof(float), nullptr, &error);
//    cl_mem cBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, count * sizeof(float), nullptr, &error);
//
//    if(adjustedSize != size)
//    {
//        float zero = 0.0f;
//        error = clEnqueueFillBuffer(queue, aBuffer, &zero, sizeof(float), 0, count * sizeof(float), 0, nullptr, nullptr);
//        error = clEnqueueFillBuffer(queue, bBuffer, &zero, sizeof(float), 0, count * sizeof(float), 0, nullptr, nullptr);
//
//        size_t bufferOffset[] = {0, 0, 0};
//        size_t hostOffset[] = {0, 0, 0};
//        size_t sizes[] = {size * sizeof(float), size, 1};
//        error = clEnqueueWriteBufferRect(queue, aBuffer, false, bufferOffset, hostOffset, sizes, adjustedSize * sizeof(float), 0, size * sizeof(float), 0, a, 0, nullptr, nullptr);
//        error = clEnqueueWriteBufferRect(queue, bBuffer, false, bufferOffset, hostOffset, sizes, adjustedSize * sizeof(float), 0, size * sizeof(float), 0, b, 0, nullptr, nullptr);
//    }
//    else
//    {
//        error = clEnqueueWriteBuffer(queue, aBuffer, false, 0, count * sizeof(float), a, 0, nullptr, nullptr);
//        error = clEnqueueWriteBuffer(queue, bBuffer, false, 0, count * sizeof(float), b, 0, nullptr, nullptr);
//    }
//
//    clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuffer);
//    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bBuffer);
//    clSetKernelArg(kernel, 2, sizeof(cl_mem), &cBuffer);
//    clSetKernelArg(kernel, 3, sizeof(cl_uint), &adjustedSize);
//
//    size_t adjustedWorkSize = roundToMultiple(adjustedSize, workGroupSize * elementsPerThread);
//
//    size_t globalWorkSizes[] = { adjustedWorkSize / elementsPerThread, adjustedWorkSize / elementsPerThread };
//    size_t localWorkSizes[] = { workGroupSize, workGroupSize };
//
//    error = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);
//
//    if(adjustedSize != size)
//    {
//        size_t bufferOffset[] = {0, 0, 0};
//        size_t hostOffset[] = {0, 0, 0};
//        size_t sizes[] = {size * sizeof(float), size, 1};
//        error = clEnqueueReadBufferRect(queue, cBuffer, true, bufferOffset, hostOffset, sizes, adjustedSize * sizeof(float), 0, size * sizeof(float), 0, c, 0, nullptr, nullptr);
//    }
//    else
//        error = clEnqueueReadBuffer(queue, cBuffer, true, 0, count * sizeof(float), c, 0, nullptr, nullptr);
//
//    clReleaseMemObject(aBuffer);
//    clReleaseMemObject(bBuffer);
//    clReleaseMemObject(cBuffer);
//}
//
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
    string source3 = readFile("../../../src/scan/gpu/thesis/RecrusivceScan.cl");
    string source4 = readFile("../../../src/scan/gpu/thesis/RecrusivceVecScan.cl");

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
    cl_kernel kernel2 = clCreateKernel(program2, "ScanWorkEfficient", &error);
    cl_kernel kernel3 = clCreateKernel(program3, "MultBlock", &error);
    cl_kernel kernel4 = clCreateKernel(program4, "MultBlockLocal", &error);

    // SCAN
    size_t size = 1<<20;
    int* input = new int[size];
    int* result0 = new int[size]();
    int* result1 = new int[size]();
    int* result2 = new int[size]();
    int* result3 = new int[size]();

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

    scanCL(input, result2, size, context, queue, kernel2, 256);
    if(!memcmp(result0, result2, size))
        cerr << "validation of kernel 2 failed" << endl;
    else
        cout << "kernel 2 ok" << endl;

    scanCL(input, result3, size, context, queue, kernel3, 256);
    if(!memcmp(result0, result3, size))
        cerr << "validation of kernel 3 failed" << endl;
    else
        cout << "kernel 3 ok" << endl;

    scanCL(input, result4, size, context, queue, kernel4, 256);
    if(!memcmp(result0, result4, size))
        cerr << "validation of kernel 4 failed" << endl;
    else
        cout << "kernel 4 ok" << endl;

    delete[] input;
    delete[] result0;
    delete[] result1;
    delete[] result2;
    delete[] result3;

    error = clReleaseKernel(kernel1);
    error = clReleaseKernel(kernel2);
    error = clReleaseKernel(kernel3);
    error = clReleaseKernel(kernel4);
    error = clReleaseProgram(program1);
    error = clReleaseProgram(program2);
    error = clReleaseProgram(program3);
    error = clReleaseProgram(program4);
    error = clReleaseCommandQueue(queue);
    error = clReleaseContext(context);
    error = clReleaseDevice(device);

    return 0;
}
