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

void matrixMul(float* a, float* b, float* c, size_t size)
{
    for(size_t row = 0; row < size; row++)
    {
        for(size_t col = 0; col < size; col++)
        {
            c[row * size + col] = 0;
            for(size_t i = 0; i < size; i++)
                c[row * size + col] += a[row * size + i] * b[i * size + col];
        }
    }
}

void matrixMulCL(float* a, float* b, float* c, cl_uint size, cl_context context, cl_command_queue queue, cl_kernel kernel, size_t workGroupSize)
{
    cl_int error;

    size_t count = size * size;

    cl_mem aBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, count * sizeof(float), nullptr, &error);
    cl_mem bBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, count * sizeof(float), nullptr, &error);
    cl_mem cBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, count * sizeof(float), nullptr, &error);

    error = clEnqueueWriteBuffer(queue, bBuffer, false, 0, count * sizeof(float), b, 0, nullptr, nullptr);
    error = clEnqueueWriteBuffer(queue, aBuffer, false, 0, count * sizeof(float), a, 0, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &c);
    clSetKernelArg(kernel, 3, sizeof(cl_uint), &size);

    size_t adjustedSize = roundToMultiple(size, workGroupSize);

    size_t globalWorkSizes[] = { adjustedSize, adjustedSize };
    size_t localWorkSizes[] = { workGroupSize, workGroupSize };

    error = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);

    error = clEnqueueReadBuffer(queue, cBuffer, true, 0, count * sizeof(float), c, 0, nullptr, nullptr);

    clReleaseMemObject(aBuffer);
    clReleaseMemObject(bBuffer);
    clReleaseMemObject(cBuffer);
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
    const char* source1 = readFile("Mult.cl").c_str();
    const char* source2 = readFile("MultLocal.cl").c_str();

    cl_program program1 = clCreateProgramWithSource(context, 1, &source1, nullptr, &error);
    cl_program program2 = clCreateProgramWithSource(context, 1, &source2, nullptr, &error);

    // compile the program for the device
    clBuildProgram(program1, 1, &device, "", nullptr, &error);
    clBuildProgram(program2, 1, &device, "", nullptr, &error);

    // create the kernel
    cl_kernel kernel1 = clCreateKernel(program1, "Mult", &error);
    cl_kernel kernel2 = clCreateKernel(program2, "MultLocal", &error);

    // MATRIX MUL
    int size = 1000;
    float* a = new float[size * size];
    float* b = new float[size * size];
    float* c0 = new float[size * size];
    float* c1 = new float[size * size];
    float* c2 = new float[size * size];

    generate(a, a + size * size, []()
    {
        return rand() % 100;
    });
    generate(b, b + size * size, []()
    {
        return rand() % 100;
    });

    matrixMul(a, b, c0, size);
    matrixMulCL(a, b, c1, size, context, queue, kernel1, 256); // 1D
    matrixMulCL(a, b, c2, size, context, queue, kernel2, 16); // 2D

    if(!memcmp(c0, c1, size * size * sizeof(float)))
        cerr << "validation of kernel 1 failed" << endl;
    else
        cout << "kernel 1 ok" << endl;

    if(!memcmp(c0, c1, size * size * sizeof(float)))
        cerr << "validation of kernel 2 failed" << endl;
    else
        cout << "kernel 2 ok" << endl;

    delete[] a;
    delete[] b;
    delete[] c0;
    delete[] c1;
    delete[] c2;

    error = clReleaseKernel(kernel1);
    error = clReleaseKernel(kernel2);
    error = clReleaseProgram(program1);
    error = clReleaseProgram(program2);
    error = clReleaseCommandQueue(queue);
    error = clReleaseContext(context);
    error = clReleaseDevice(device);

    return 0;
}
