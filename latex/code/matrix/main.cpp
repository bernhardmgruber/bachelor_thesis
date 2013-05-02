#include <CL/cl.h>
#include <algorithm>
#include <iostream>
#include <stdio.h>

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
    const char* source = "...";

    cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &error);

    // compile the program for the device
    clBuildProgram(program, 1, &device, "", nullptr, &error);

    // create the kernel
    cl_kernel kernel = clCreateKernel(program, "...", &error);

    // MATRIX MUL
    int size = 1000;
    float* a = new float[size * size];
    float* b = new float[size * size];
    float* c = new float[size * size];
    float* d = new float[size * size];

    generate(a, a + size * size, []()
    {
        return rand() % 100;
    });
    generate(b, b + size * size, []()
    {
        return rand() % 100;
    });

    matrixMul(a, b, c, size);
    matrixMulCL(a, b, d, size, context, queue, kernel, 256);

    if(!memcmp(c, d, size * size * sizeof(float)))
        cerr << "validation failed" << endl;

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] d;

    error = clReleaseKernel(kernel);
    error = clReleaseProgram(program);
    error = clReleaseCommandQueue(queue);
    error = clReleaseContext(context);
    error = clReleaseDevice(device);

    return 0;
}
