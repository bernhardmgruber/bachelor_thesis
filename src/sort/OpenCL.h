#ifndef OPENCL_H
#define OPENCL_H

#include <CL/CL.h>
#include <string>
#include <vector>

using namespace std;

class Context;
class CommandQueue;
class Program;
class Kernel;
class Buffer;

/**
 * Singleton.
 */
class OpenCL
{
    public:
        static void init();
        static void cleanup();
        static Context* getGPUContext();
        static Context* getCPUContext();

    private:
        OpenCL();
        virtual ~OpenCL();

        static cl_platform_id platform;

        static vector<Context*> contexts;
};

class Context
{
    public:
        Context(cl_context context, cl_device_id device);
        virtual ~Context();

        Program* createProgram(string source);
        CommandQueue* createCommandQueue();
        Buffer* createBuffer(cl_mem_flags flags, size_t size, void* ptr = nullptr);

    private:
        string readFile(string fileName);

        cl_device_id device;
        cl_context context;

        vector<Program*> programs;
        vector<CommandQueue*> queues;
        vector<Buffer*> buffers;
};

class Program
{
    public:
        Program(cl_program program);
        virtual ~Program();

        Kernel* createKernel(string entry);

    private:
        cl_program program;

        vector<Kernel*> kernels;
};

class Kernel
{
    public:
        Kernel(cl_kernel kernel);

        void setArg(cl_uint index, size_t size, const void* value);

    private:
        cl_kernel kernel;

    friend CommandQueue;
};

class CommandQueue
{
    public:
        CommandQueue(cl_command_queue queue);

        void enqueueKernel(Kernel kernel, cl_uint dimension);
        void enqueueRead(Buffer& buffer, void* destination, size_t offset, size_t size, bool blocking = true);
        void enqueueRead(Buffer& buffer, void* destination, bool blocking = true);

    private:
        cl_command_queue queue;
};

class Buffer
{
    public:
        Buffer(cl_mem buffer, size_t size);

    private:
        cl_mem buffer;
        size_t size;

    friend CommandQueue;
};

#endif // OPENCL_H
