#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>

#include "OpenCL.h"

using namespace std;

// static variables
cl_platform_id OpenCL::platform;
//vector<Context*> OpenCL::contexts;

static cl_int error;

static void checkError(int line)
{
    if(error != CL_SUCCESS)
    {
        stringstream ss;
        ss << "Error at line " << line << ": " << error;
        throw OpenCLException(ss.str());
    }

}

//
// class OpenCLException
//
OpenCLException::OpenCLException(string msg)
    : msg(msg)
{

}

const char* OpenCLException::what() throw()
{
    return msg.c_str();
}

//
// class OpenCL
//

void OpenCL::init()
{
    // get the first available platform
    error = clGetPlatformIDs(1, &platform, nullptr);
    checkError(__LINE__);
}

void OpenCL::cleanup()
{
    //for(Context* c : contexts)
    //    delete c;
    //contexts.clear();
}

Context* OpenCL::getGPUContext()
{
    // get a GPU from this platform
    cl_device_id device;
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    checkError(__LINE__);

    // create a context to work with OpenCL
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
    checkError(__LINE__);

    // create Context object
    Context* contextObj = new Context(context, device);
    //contexts.push_back(contextObj);

    return contextObj;
}

Context* OpenCL::getCPUContext()
{
    // get a GPU from this platform
    cl_device_id device;
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
    checkError(__LINE__);

    // create a context to work with OpenCL
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
    checkError(__LINE__);

    // create Context object
    Context* contextObj = new Context(context, device);
    //contexts.push_back(contextObj);

    return contextObj;
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
    //for(Program* p : programs)
    //    delete p;
    //programs.clear();
    //for(CommandQueue* q : queues)
    //    delete q;
    //queues.clear();
    //for(Buffer* b : buffers)
    //    delete b;
    //buffers.clear();

    clReleaseContext(context);
    clReleaseDevice(device);
}

Program* Context::createProgram(string sourceFile)
{
    string sourceString = readFile(sourceFile);
    const char* source = sourceString.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &error);
    checkError(__LINE__);

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

        throw OpenCLException(log);
    }

    // create Program object
    Program* programObj = new Program(program);
    //programs.push_back(programObj);

    return programObj;
}

CommandQueue* Context::createCommandQueue()
{
    // create a new command queue, where kernels can be executed
    cl_command_queue cmdqueue = clCreateCommandQueue(context, device, 0, &error);
    checkError(__LINE__);

    // create CommandQueue object
    CommandQueue* queueObj = new CommandQueue(cmdqueue);
    //queues.push_back(queueObj);

    return queueObj;
}

Buffer* Context::createBuffer(cl_mem_flags flags, size_t size, void* ptr)
{
    cl_mem buffer = clCreateBuffer(context, flags, size, ptr, &error);
    checkError(__LINE__);


    // create CommandQueue object
    Buffer* bufferObj = new Buffer(buffer, size);
    //buffers.push_back(bufferObj);

    return bufferObj;
}


string Context::readFile(string fileName)
{
    ifstream file(fileName, ios::in);
    if(!file)
        throw OpenCLException("Error opening file " + fileName);

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

Program::~Program()
{
    //for(Kernel* k : kernels)
    //    delete k;
    //kernels.clear();

    clReleaseProgram(program);
}

Kernel* Program::createKernel(string entry)
{
    cl_kernel kernel = clCreateKernel(program, entry.c_str(), &error);
    checkError(__LINE__);

    // create Kernel object
    Kernel* kernelObj = new Kernel(kernel);
//    kernels.push_back(kernelObj);

    return kernelObj;
}

//
// class Kernel
//

Kernel::Kernel(cl_kernel kernel)
    : kernel(kernel)
{
}

Kernel::~Kernel()
{
    clReleaseKernel(kernel);
}

void Kernel::setArg(cl_uint index, Buffer* buffer)
{
    clSetKernelArg(kernel, index, sizeof(cl_mem), (const void*)&buffer->buffer);
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

CommandQueue::~CommandQueue()
{
    clReleaseCommandQueue(queue);
}

void CommandQueue::enqueueKernel(Kernel* kernel, cl_uint dimension, const size_t* globalWorkSizes)
{
    error = clEnqueueNDRangeKernel(queue, kernel->kernel, dimension, nullptr, globalWorkSizes, nullptr, 0, nullptr, nullptr);
    checkError(__LINE__);
}

void CommandQueue::enqueueRead(Buffer* buffer, void* destination, size_t offset, size_t size, bool blocking)
{
    error = clEnqueueReadBuffer(queue, buffer->buffer, blocking, offset, size, destination, 0, nullptr, nullptr);
    checkError(__LINE__);
}

void CommandQueue::enqueueRead(Buffer* buffer, void* destination, bool blocking)
{
    error = clEnqueueReadBuffer(queue, buffer->buffer, blocking, 0, buffer->size, destination, 0, nullptr, nullptr);
    checkError(__LINE__);
}

void CommandQueue::enqueueWrite(Buffer* buffer, const void* source, bool blocking)
{
    error = clEnqueueWriteBuffer(queue, buffer->buffer, blocking, 0, buffer->size, source, 0, nullptr, nullptr);
    checkError(__LINE__);
}

void CommandQueue::finish()
{
    error = clFinish(queue);
    checkError(__LINE__);
}

//
// class Bufffer
//

Buffer::Buffer(cl_mem buffer, size_t size)
    : buffer(buffer), size(size)
{

}

Buffer::~Buffer()
{
    clReleaseMemObject(buffer);
}
