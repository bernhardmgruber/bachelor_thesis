#include <CL/cl.h>
#include <stdio.h>

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
    const char* source = ""
    "__kernel void vectorAdd(const __global float* a, const __global float* b, __global float* c)"
    "{"
    "    size_t id = get_global_id(0);"
    "    c[id] = a[id] + b[id];"
    "}";

    cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &error);

    // compile the program for the device
    error = clBuildProgram(program, 1, &device, "", nullptr, nullptr);

    // create the kernel
    cl_kernel kernel = clCreateKernel(program, "vectorAdd", &error);

    // input data
    size_t length = 1024;
    float* a = (float*)malloc(length * sizeof(float));
    float* b = (float*)malloc(length * sizeof(float));

    for(size_t i = 0; i < length; i++)
    {
        a[i] = i;
        b[i] = length - i;
    }

    // create input buffers
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, length * sizeof(float), nullptr, &error);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, length * sizeof(float), nullptr, &error);

    // write input data to buffers
    error = clEnqueueWriteBuffer(queue, bufferA, false, 0, length * sizeof(float), a, 0, nullptr, nullptr);
    error = clEnqueueWriteBuffer(queue, bufferB, false, 0, length * sizeof(float), b, 0, nullptr, nullptr);

    // create output buffer
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, length * sizeof(float), nullptr, &error);

    // set the kernel's arguments
    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);

    // run the kernel
    error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &length, nullptr, 0, nullptr, nullptr);

    // allocate memory for output
    float* c = (float*)malloc(length * sizeof(float));

    // read result back to host, block until data has been copied
    error = clEnqueueReadBuffer(queue, bufferC, true, 0, length * sizeof(float), c, 0, nullptr, nullptr);

    // output result
    for(size_t i = 0; i < length; i++)
        printf("%.1f,", c[i]);

    // cleanup
    free(a);
    free(b);
    free(c);

    error = clReleaseMemObject(bufferA);
    error = clReleaseMemObject(bufferB);
    error = clReleaseMemObject(bufferC);
    error = clReleaseKernel(kernel);
    error = clReleaseProgram(program);
    error = clReleaseCommandQueue(queue);
    error = clReleaseContext(context);
    error = clReleaseDevice(device);

    return 0;
}
