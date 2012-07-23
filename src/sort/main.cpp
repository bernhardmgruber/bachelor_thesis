#include <CL/CL.h>
#include <iostream>
#include <fstream>

#include "SortingAlgorithm.h"
#include "Quicksort.h"

using namespace std;

/*cl_command_queue cmdqueue;
cl_kernel kernel;
cl_mem srcMemory;
cl_mem destMemory;
cl_program program;
cl_context context;

bool ReadFile(const char* fileName, string& buffer)
{
    ifstream sourceFile(fileName, ios::binary | ios::in);
    if(!sourceFile)
    {
        cerr << "Error opening file " << fileName << endl;
        return false;
    }

    buffer = string((istreambuf_iterator<char>(sourceFile)), istreambuf_iterator<char>());

    sourceFile.close();

    return true;
}

#define CHECK(error) if(error != CL_SUCCESS) { cout << "Error at line " << __LINE__ << ": " << error << endl; }*/

int main()
{
    SortingAlgorithm<int, 10000000>* alg = new Quicksort<int, 10000000>();
    alg->runTest();
    delete alg;


    /*cl_int error;

    // get the first available platform
    cl_platform_id platform;
    error = clGetPlatformIDs(1, &platform, nullptr);
    CHECK(error);

    // get a GPU from this platform
    cl_device_id device;
    #ifdef CL_GPU
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    #else
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
    #endif
    CHECK(error);

    // create a context to work with OpenCL
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
    CHECK(error);

    // read the kernel source
    ifstream sourceFile("gameoflife.cl");
    string buffer((istreambuf_iterator<char>(sourceFile)), istreambuf_iterator<char>());
    sourceFile.close();
    const char* source = buffer.c_str();

    // create an OpenCL program from the source code
    program = clCreateProgramWithSource(context, 1, &source, nullptr, &error);
    CHECK(error);

    // build the program
    error = clBuildProgram(program, 1, &device, "-w", nullptr, nullptr);
    if(error != CL_SUCCESS)
    {
        // get the error log size
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

        // allocate enough space and get the log
        char* log = new char[logSize + 1];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, nullptr);
        log[logSize] = '\0';

        // print the build log and delete it
        cout << log << endl;
        delete[] log;

        return false;
    }

    // set the entry point for an OpenCL kernel
    kernel = clCreateKernel(program, "NextGeneration", &error);
    CHECK(error);

    // create a buffer for the grid data and init it with client data
    srcMemory = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dataWidth * dataHeight * sizeof(bool), data, &error);
    CHECK(error);

    destMemory = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataWidth * dataHeight * sizeof(bool), nullptr, &error);
    CHECK(error);

    // set the created buffer as argument to the kernel
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &srcMemory);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &destMemory);

    // create a new command queue, where kernels can be executed
    cmdqueue = clCreateCommandQueue(context, device, 0, &error);
    CHECK(error);

    error = clEnqueueNDRangeKernel(cmdqueue, kernel, 2, nullptr, globalWorkSizes, nullptr, 0, nullptr, nullptr);
    CHECK(error);

    // Copy new generation to source buffer for next run
    clEnqueueCopyBuffer(cmdqueue, destMemory, srcMemory, 0, 0, dataWidth * dataHeight * sizeof(bool), 0, nullptr, nullptr);

    clEnqueueReadBuffer(cmdqueue, destMemory, CL_TRUE, 0, dataWidth * dataHeight * sizeof(bool), data, 0, nullptr, nullptr);

    clReleaseMemObject(srcMemory);
    clReleaseMemObject(destMemory);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdqueue);
    clReleaseContext(context);*/

    return 0;
}
