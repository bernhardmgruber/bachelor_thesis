#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>

#include "OpenCL.h"

using namespace std;

// static variables
cl_platform_id OpenCL::platform;
//vector<Context*> OpenCL::contexts;

static const int errorCodesCount = 63;

static const char* errorCodes[errorCodesCount] = {
  "CL_SUCCESS", // 0
  "CL_DEVICE_NOT_FOUND", // -1
  "CL_DEVICE_NOT_AVAILABLE", // -2
  "CL_COMPILER_NOT_AVAILABLE", // -3
  "CL_MEM_OBJECT_ALLOCATION_FAILURE", // -4
  "CL_OUT_OF_RESOURCES", // -5
  "CL_OUT_OF_HOST_MEMORY", // -6
  "CL_PROFILING_INFO_NOT_AVAILABLE", // -7
  "CL_MEM_COPY_OVERLAP", // -8
  "CL_IMAGE_FORMAT_MISMATCH", // -9
  "CL_IMAGE_FORMAT_NOT_SUPPORTED", // -10
  "CL_BUILD_PROGRAM_FAILURE", // -11
  "CL_MAP_FAILURE", // -12
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // -13..-29
  "CL_INVALID_VALUE", // -30
  "CL_INVALID_DEVICE_TYPE", // -31
  "CL_INVALID_PLATFORM", // -32
  "CL_INVALID_DEVICE", // -33
  "CL_INVALID_CONTEXT", // -34
  "CL_INVALID_QUEUE_PROPERTIES", // -35
  "CL_INVALID_COMMAND_QUEUE", // -36
  "CL_INVALID_HOST_PTR", // -37
  "CL_INVALID_MEM_OBJECT", // -38
  "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR", // -39
  "CL_INVALID_IMAGE_SIZE", // -40
  "CL_INVALID_SAMPLER", // -41
  "CL_INVALID_BINARY", // -42
  "CL_INVALID_BUILD_OPTIONS", // -43
  "CL_INVALID_PROGRAM", // -44
  "CL_INVALID_PROGRAM_EXECUTABLE", // -45
  "CL_INVALID_KERNEL_NAME", // -46
  "CL_INVALID_KERNEL_DEFINITION", // -47
  "CL_INVALID_KERNEL", // -48
  "CL_INVALID_ARG_INDEX", // -49
  "CL_INVALID_ARG_VALUE", // -50
  "CL_INVALID_ARG_SIZE", // -51
  "CL_INVALID_KERNEL_ARGS", // -52
  "CL_INVALID_WORK_DIMENSION", // -53
  "CL_INVALID_WORK_GROUP_SIZE", // -54
  "CL_INVALID_WORK_ITEM_SIZE", // -55
  "CL_INVALID_GLOBAL_OFFSET", // -56
  "CL_INVALID_EVENT_WAIT_LIST", // -57
  "CL_INVALID_EVENT", // -58
  "CL_INVALID_OPERATION", // -59
  "CL_INVALID_GL_OBJECT", // -60
  "CL_INVALID_BUFFER_SIZE", // -61
  "CL_INVALID_MIP_LEVEL" // -62
};


static cl_int error;

void checkError(int line)
{
    if(error != CL_SUCCESS)
    {
        stringstream ss;
        ss << "Error at line " << line << ": ";
        if(-error < errorCodesCount)
            ss << errorCodes[-error] << " (" << error << ")";
        else
            ss << error;
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
    //clReleaseDevice(device);
}

Program* Context::createProgram(string sourceFile, string options)
{
    string sourceString = readFile(sourceFile);
    const char* source = sourceString.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &error);
    checkError(__LINE__);

    // build the program
    error = clBuildProgram(program, 1, &device, ("-w " + options).c_str(), nullptr, nullptr);
    if(error != CL_SUCCESS)
    {
        // get the error log size
        size_t logSize;
        error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        checkError(__LINE__);

        // allocate enough space and get the log
        char* log = new char[logSize + 1];
        error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, nullptr);
        checkError(__LINE__);
        log[logSize] = '\0';

        string logStr = string(log);
        delete[] log;

        // print the build log and delete it
        cerr << logStr << endl;

        throw OpenCLException(logStr);
    }

    // create Program object
    Program* programObj = new Program(program, this);
    //programs.push_back(programObj);

    return programObj;
}

CommandQueue* Context::createCommandQueue()
{
    // create a new command queue, where kernels can be executed
    cl_command_queue cmdqueue = clCreateCommandQueue(context, device, 0, &error);
    checkError(__LINE__);

    // create CommandQueue object
    CommandQueue* queueObj = new CommandQueue(cmdqueue, this);
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

size_t Context::getInfoSize(cl_device_info info)
{
    size_t value;
    clGetDeviceInfo(device, info, sizeof(value), (void*) &value, nullptr);

    return value;
}

string Context::getInfoString(cl_device_info info)
{
    size_t size = 0;
    clGetDeviceInfo(device, info, 0, nullptr, &size);

    char* buffer = new char[size + 1];
    clGetDeviceInfo(device, info, size + 1, buffer, nullptr);
    buffer[size + 1] = 0;

    return string(buffer);
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

Program::Program(cl_program program, Context* context)
    : program(program), context(context)
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
    Kernel* kernelObj = new Kernel(kernel, context);
//    kernels.push_back(kernelObj);

    return kernelObj;
}

//
// class Kernel
//

Kernel::Kernel(cl_kernel kernel, Context* context)
    : kernel(kernel), context(context)
{
}

Kernel::~Kernel()
{
    error = clReleaseKernel(kernel);
    checkError(__LINE__);
}

void Kernel::setArg(cl_uint index, Buffer* buffer)
{
    error = clSetKernelArg(kernel, index, sizeof(cl_mem), (const void*)&buffer->buffer);
    checkError(__LINE__);
}

void Kernel::setArg(cl_uint index, size_t size, const void* value)
{
    error = clSetKernelArg(kernel, index, size, value);
    checkError(__LINE__);
}

size_t Kernel::getWorkGroupSize()
{
    size_t size;
    error = clGetKernelWorkGroupInfo (kernel, context->device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &size, nullptr);
    checkError(__LINE__);
    return size;
}

//
// class CommandQueue
//

CommandQueue::CommandQueue(cl_command_queue queue ,Context* context)
    : queue(queue), context(context)
{
}

CommandQueue::~CommandQueue()
{
    clReleaseCommandQueue(queue);
}

void CommandQueue::enqueueKernel(Kernel* kernel, cl_uint dimension, const size_t* globalWorkSizes, const size_t* localWorkSizes)
{
    error = clEnqueueNDRangeKernel(queue, kernel->kernel, dimension, nullptr, globalWorkSizes, localWorkSizes, 0, nullptr, nullptr);
    checkError(__LINE__);
}

void CommandQueue::enqueueTask(Kernel* kernel)
{
    error = clEnqueueTask(queue, kernel->kernel, 0, nullptr, nullptr);
    checkError(__LINE__);
}

void CommandQueue::enqueueRead(Buffer* buffer, void* destination, bool blocking)
{
    enqueueRead(buffer, destination, 0, buffer->size, blocking);
}

void CommandQueue::enqueueRead(Buffer* buffer, void* destination, size_t offset, size_t size, bool blocking)
{
    error = clEnqueueReadBuffer(queue, buffer->buffer, blocking, offset, size, destination, 0, nullptr, nullptr);
    checkError(__LINE__);
}

void CommandQueue::enqueueWrite(Buffer* buffer, const void* source, bool blocking)
{
    error = clEnqueueWriteBuffer(queue, buffer->buffer, blocking, 0, buffer->size, source, 0, nullptr, nullptr);
    checkError(__LINE__);
}

void CommandQueue::enqueueCopy(Buffer* src, Buffer* dest)
{
    enqueueCopy(src, dest, 0, 0, dest->size);
}

void CommandQueue::enqueueCopy(Buffer* src, Buffer* dest, size_t srcOffset, size_t destOffset, size_t size)
{
    error = clEnqueueCopyBuffer(queue, src->buffer, dest->buffer, srcOffset, destOffset, size, 0, nullptr, nullptr);
    checkError(__LINE__);
}

void CommandQueue::enqueueBarrier()
{
    error = clEnqueueBarrier(queue);
    checkError(__LINE__);
}

void CommandQueue::flush()
{
    error = clFlush(queue);
    checkError(__LINE__);
}

void CommandQueue::finish()
{
    error = clFinish(queue);
    checkError(__LINE__);
}

Context* CommandQueue::getContext()
{
    return context;
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

size_t Buffer::getSize()
{
    return size;
}
