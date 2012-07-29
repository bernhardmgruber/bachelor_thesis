#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>

#include "OpenCL.h"

using namespace std;

// static variables
cl_platform_id OpenCL::platform;

static cl_int error;

static void checkError()
{
    if(error != CL_SUCCESS)
    {
        stringstream ss;
        ss << "Error at line " << __LINE__ << ": " << error;
        throw runtime_error(ss.str());
    }

}


//
// class OpenCL
//

void OpenCL::init()
{
    // get the first available platform
    error = clGetPlatformIDs(1, &platform, nullptr);
    checkError();
}

Context OpenCL::getGPUContext()
{
    // get a GPU from this platform
    cl_device_id device;
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    checkError();

    // create a context to work with OpenCL
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
    checkError();

    return Context(context, device);
}

Context OpenCL::getCPUContext()
{
    // get a GPU from this platform
    cl_device_id device;
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
    checkError();

    // create a context to work with OpenCL
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
    checkError();

    return Context(context, device);
}

OpenCL::OpenCL()
{
}

OpenCL::~OpenCL()
{
}

//
// class Context
//

Context::Context(cl_context context, cl_device_id device)
    : device(device), context(context)
{
}

Context::~Context()
{
}

Program Context::createProgram(string sourceFile)
{

    const char* source = readFile(sourceFile).c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &error);
    checkError();

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

        throw runtime_error(log);
    }

    return Program(program);
}

CommandQueue Context::createCommandQueue()
{
    // create a new command queue, where kernels can be executed
    cl_command_queue cmdqueue = clCreateCommandQueue(context, device, 0, &error);
    checkError();

    return CommandQueue(cmdqueue);
}

Buffer Context::createBuffer(cl_mem_flags flags, size_t size, void* ptr)
{
    cl_mem buffer = clCreateBuffer(context, flags, size, ptr, &error);
    checkError();

    return Buffer(buffer, size);
}


string Context::readFile(string fileName)
{
    ifstream file(fileName, ios::in);
    if(!file)
    {
        cerr << "Error opening file " << fileName << endl;
        throw runtime_error("LOL");
    }

    string buffer = string((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());

    file.close();

    return buffer;
}

//
// class Program
//

Program::Program(cl_program program)
    : program(program)
{
}

Kernel Program::createKernel(string entry)
{
    cl_kernel kernel = clCreateKernel(program, entry.c_str(), &error);
    checkError();

    return Kernel(kernel);
}

//
// class Kernel
//

Kernel::Kernel(cl_kernel kernel)
    : kernel(kernel)
{



}

void Kernel::setArg(cl_uint index, size_t size, const void* value)
{
    clSetKernelArg(kernel, index, size, value);
}

//
// class CommandQueue
//

CommandQueue::CommandQueue(cl_command_queue queue)
    : queue(queue)
{
}

void CommandQueue::enqueueKernel(Kernel kernel, cl_uint dimension)
{
    error = clEnqueueNDRangeKernel(queue, kernel.kernel, dimension, nullptr, nullptr, nullptr, 0, nullptr, nullptr);
    checkError();
}

void CommandQueue::enqueueRead(Buffer& buffer, void* destination, size_t offset, size_t size, bool blocking)
{
    error = clEnqueueReadBuffer(queue, buffer.buffer, blocking, offset, size, destination, 0, nullptr, nullptr);
    checkError();
}

void CommandQueue::enqueueRead(Buffer& buffer, void* destination, bool blocking)
{
    error = clEnqueueReadBuffer(queue, buffer.buffer, blocking, 0, buffer.size, destination, 0, nullptr, nullptr);
    checkError();
}

//
// class Bufffer
//

Buffer::Buffer(cl_mem buffer, size_t size)
    : buffer(buffer), size(size)
{

}
