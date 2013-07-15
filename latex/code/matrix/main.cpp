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

    error = clEnqueueWriteBuffer(queue, aBuffer, false, 0, count * sizeof(float), a, 0, nullptr, nullptr);
    error = clEnqueueWriteBuffer(queue, bBuffer, false, 0, count * sizeof(float), b, 0, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &cBuffer);
    clSetKernelArg(kernel, 3, sizeof(cl_uint), &size);

    size_t adjustedWorkSize = roundToMultiple(size, workGroupSize);

    size_t globalWorkSizes[] = { adjustedWorkSize, adjustedWorkSize };
    size_t localWorkSizes[] = { workGroupSize, workGroupSize };

    error = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);

    error = clEnqueueReadBuffer(queue, cBuffer, true, 0, count * sizeof(float), c, 0, nullptr, nullptr);

    clReleaseMemObject(aBuffer);
    clReleaseMemObject(bBuffer);
    clReleaseMemObject(cBuffer);
}

#define TILE_SIZE 16

void matrixMulCLLocal(float* a, float* b, float* c, cl_uint size, cl_context context, cl_command_queue queue, cl_kernel kernel)
{
    cl_int error;

    cl_uint adjustedSize = roundToMultiple(size, TILE_SIZE);

    size_t count = adjustedSize * adjustedSize;

    cl_mem aBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, count * sizeof(float), nullptr, &error);
    cl_mem bBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, count * sizeof(float), nullptr, &error);
    cl_mem cBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, count * sizeof(float), nullptr, &error);

    if(adjustedSize != size)
    {
        float zero = 0.0f;
        error = clEnqueueFillBuffer(queue, aBuffer, &zero, sizeof(float), 0, count * sizeof(float), 0, nullptr, nullptr);
        error = clEnqueueFillBuffer(queue, bBuffer, &zero, sizeof(float), 0, count * sizeof(float), 0, nullptr, nullptr);

        size_t bufferOffset[] = {0, 0, 0};
        size_t hostOffset[] = {0, 0, 0};
        size_t sizes[] = {size * sizeof(float), size, 1};
        error = clEnqueueWriteBufferRect(queue, aBuffer, false, bufferOffset, hostOffset, sizes, adjustedSize * sizeof(float), 0, size * sizeof(float), 0, a, 0, nullptr, nullptr);
        error = clEnqueueWriteBufferRect(queue, bBuffer, false, bufferOffset, hostOffset, sizes, adjustedSize * sizeof(float), 0, size * sizeof(float), 0, b, 0, nullptr, nullptr);
    }
    else
    {
        error = clEnqueueWriteBuffer(queue, aBuffer, false, 0, count * sizeof(float), a, 0, nullptr, nullptr);
        error = clEnqueueWriteBuffer(queue, bBuffer, false, 0, count * sizeof(float), b, 0, nullptr, nullptr);
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &cBuffer);
    clSetKernelArg(kernel, 3, sizeof(cl_uint), &adjustedSize);

    size_t globalWorkSizes[] = { adjustedSize, adjustedSize };
    size_t localWorkSizes[] = { TILE_SIZE, TILE_SIZE };

    error = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);

    if(adjustedSize != size)
    {
        size_t bufferOffset[] = {0, 0, 0};
        size_t hostOffset[] = {0, 0, 0};
        size_t sizes[] = {size * sizeof(float), size, 1};
        error = clEnqueueReadBufferRect(queue, cBuffer, true, bufferOffset, hostOffset, sizes, adjustedSize * sizeof(float), 0, size * sizeof(float), 0, c, 0, nullptr, nullptr);
    }
    else
        error = clEnqueueReadBuffer(queue, cBuffer, true, 0, count * sizeof(float), c, 0, nullptr, nullptr);

    clReleaseMemObject(aBuffer);
    clReleaseMemObject(bBuffer);
    clReleaseMemObject(cBuffer);
}

#define BLOCK_SIZE 4

void matrixMulCLRect(float* a, float* b, float* c, cl_uint size, cl_context context, cl_command_queue queue, cl_kernel kernel, size_t workGroupSize, int multiple, int elementsPerThread)
{
    cl_int error;

    cl_uint adjustedSize = roundToMultiple(size, multiple);

    size_t count = adjustedSize * adjustedSize;

    cl_mem aBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, count * sizeof(float), nullptr, &error);
    cl_mem bBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, count * sizeof(float), nullptr, &error);
    cl_mem cBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, count * sizeof(float), nullptr, &error);

    if(adjustedSize != size)
    {
        float zero = 0.0f;
        error = clEnqueueFillBuffer(queue, aBuffer, &zero, sizeof(float), 0, count * sizeof(float), 0, nullptr, nullptr);
        error = clEnqueueFillBuffer(queue, bBuffer, &zero, sizeof(float), 0, count * sizeof(float), 0, nullptr, nullptr);

        size_t bufferOffset[] = {0, 0, 0};
        size_t hostOffset[] = {0, 0, 0};
        size_t sizes[] = {size * sizeof(float), size, 1};
        error = clEnqueueWriteBufferRect(queue, aBuffer, false, bufferOffset, hostOffset, sizes, adjustedSize * sizeof(float), 0, size * sizeof(float), 0, a, 0, nullptr, nullptr);
        error = clEnqueueWriteBufferRect(queue, bBuffer, false, bufferOffset, hostOffset, sizes, adjustedSize * sizeof(float), 0, size * sizeof(float), 0, b, 0, nullptr, nullptr);
    }
    else
    {
        error = clEnqueueWriteBuffer(queue, aBuffer, false, 0, count * sizeof(float), a, 0, nullptr, nullptr);
        error = clEnqueueWriteBuffer(queue, bBuffer, false, 0, count * sizeof(float), b, 0, nullptr, nullptr);
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &cBuffer);
    clSetKernelArg(kernel, 3, sizeof(cl_uint), &adjustedSize);

    size_t adjustedWorkSize = roundToMultiple(adjustedSize, workGroupSize * elementsPerThread);

    size_t globalWorkSizes[] = { adjustedWorkSize / elementsPerThread, adjustedWorkSize / elementsPerThread };
    size_t localWorkSizes[] = { workGroupSize, workGroupSize };

    error = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);

    if(adjustedSize != size)
    {
        size_t bufferOffset[] = {0, 0, 0};
        size_t hostOffset[] = {0, 0, 0};
        size_t sizes[] = {size * sizeof(float), size, 1};
        error = clEnqueueReadBufferRect(queue, cBuffer, true, bufferOffset, hostOffset, sizes, adjustedSize * sizeof(float), 0, size * sizeof(float), 0, c, 0, nullptr, nullptr);
    }
    else
        error = clEnqueueReadBuffer(queue, cBuffer, true, 0, count * sizeof(float), c, 0, nullptr, nullptr);

    clReleaseMemObject(aBuffer);
    clReleaseMemObject(bBuffer);
    clReleaseMemObject(cBuffer);
}

#if 0
#define BLOCK_SIZE 4
    ...
    cl_uint adjustedSize = roundToMultiple(size, BLOCK_SIZE);
    ...
    size_t adjustedWorkSize = roundToMultiple(adjustedSize, workGroupSize * BLOCK_SIZE);

    size_t globalWorkSizes[] = { adjustedWorkSize / BLOCK_SIZE, adjustedWorkSize / BLOCK_SIZE };
    size_t localWorkSizes[] = { workGroupSize, workGroupSize };
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

bool memcmp_float(const float* a, const float* b, size_t count, float epsilon = 0.001)
{
    for(size_t i = 0; i < count; i++)
        if(fabs(a[i] - b[i]) > epsilon)
            return false;

    return true;
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
    string source1 = readFile("Mult.cl");
    string source2 = readFile("MultLocal.cl");
    string source3 = readFile("MultBlock.cl");
    string source4 = readFile("MultBlockLocal.cl");

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
    cl_kernel kernel1 = clCreateKernel(program1, "Mult", &error);
    cl_kernel kernel2 = clCreateKernel(program2, "MultLocal", &error);
    cl_kernel kernel3 = clCreateKernel(program3, "MultBlock", &error);
    cl_kernel kernel4 = clCreateKernel(program4, "MultBlockLocal", &error);

    // MATRIX MUL
    int size = 1000;
    float* a = new float[size * size];
    float* b = new float[size * size];
    float* c0 = new float[size * size]();
    float* c1 = new float[size * size]();
    float* c2 = new float[size * size]();
    float* c3 = new float[size * size]();
    float* c4 = new float[size * size]();

    generate(a, a + size * size, []()
    {
        return rand() % 100;
    });
    generate(b, b + size * size, []()
    {
        return rand() % 100;
    });

    matrixMul(a, b, c0, size);

    matrixMulCL(a, b, c1, size, context, queue, kernel1, 16); // 2D
    if(!memcmp_float(c0, c1, size * size))
        cerr << "validation of kernel 1 failed" << endl;
    else
        cout << "kernel 1 ok" << endl;

    matrixMulCLLocal(a, b, c2, size, context, queue, kernel2); // 2D
    if(!memcmp_float(c0, c2, size * size))
        cerr << "validation of kernel 2 failed" << endl;
    else
        cout << "kernel 2 ok" << endl;

    matrixMulCLRect(a, b, c3, size, context, queue, kernel3, 16, BLOCK_SIZE, BLOCK_SIZE); // 2D
    if(!memcmp_float(c0, c3, size * size))
        cerr << "validation of kernel 3 failed" << endl;
    else
        cout << "kernel 3 ok" << endl;

    matrixMulCLRect(a, b, c4, size, context, queue, kernel4, 16, BLOCK_SIZE * TILE_SIZE, BLOCK_SIZE); // 2D
    if(!memcmp_float(c0, c4, size * size))
        cerr << "validation of kernel 4 failed" << endl;
    else
        cout << "kernel 4 ok" << endl;

    delete[] a;
    delete[] b;
    delete[] c0;
    delete[] c1;
    delete[] c2;
    delete[] c3;
    delete[] c4;

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
