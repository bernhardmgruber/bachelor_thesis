#ifndef OPENCL_H
#define OPENCL_H

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/CL.h>
#include <string>
#include <vector>
#include <exception>

using namespace std;

void checkError(int line);

class Context;
class CommandQueue;
class Program;
class Kernel;
class Buffer;

class OpenCLException : public exception
{
    public:
        OpenCLException(string msg);

        virtual const char* what() throw();

    private:
        string msg;
};

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

        //static vector<Context*> contexts;
};

/**
 * Represents a device and a context opened at this device.
 */
class Context
{
    public:
        Context(cl_context context, cl_device_id device);
        virtual ~Context();

        Program* createProgram(string source, string options = "");
        CommandQueue* createCommandQueue();
        Buffer* createBuffer(cl_mem_flags flags, size_t size, void* ptr = nullptr);

        size_t getInfoSize(cl_device_info info);
        string getInfoString(cl_device_info info);

    private:
        string readFile(string fileName);

        cl_device_id device;
        cl_context context;

        //vector<Program*> programs;
        //vector<CommandQueue*> queues;
        //vector<Buffer*> buffers;
        friend Kernel;
};

class Program
{
    public:
        Program(cl_program program, Context* context);
        virtual ~Program();

        Kernel* createKernel(string entry);

    private:
        cl_program program;

        /// The context and device this program has been created for.
        Context* context;
};

class Kernel
{
    public:
        Kernel(cl_kernel kernel, Context* context);
        ~Kernel();

        void setArg(cl_uint index, Buffer* buffer);
        void setArg(cl_uint index, size_t size, const void* value);

        template <typename T>
        void setArg(cl_uint index, T value)
        {
            clSetKernelArg(kernel, index, sizeof(T), &value);
            checkError(__LINE__);
        }

        size_t getWorkGroupSize();

    private:
        cl_kernel kernel;

        /// The context and device this kernel has been created for.
        Context* context;

    friend CommandQueue;
};

class CommandQueue
{
    public:
        CommandQueue(cl_command_queue queue);
        virtual ~CommandQueue();

        void enqueueKernel(Kernel* kernel, cl_uint dimension, const size_t* globalWorkSizes, const size_t* localWorkSizes = nullptr);
        void enqueueRead(Buffer* buffer, void* destination, bool blocking = true);
        void enqueueRead(Buffer* buffer, void* destination, size_t offset, size_t size, bool blocking = true);
        void enqueueWrite(Buffer* buffer, const void* source, bool blocking = true);
        void enqueueCopy(Buffer* src, Buffer* dest);
        void enqueueCopy(Buffer* src, Buffer* dest, size_t srcOffset, size_t destOffset, size_t size);
        void enqueueBarrier();

        void finish();

    private:
        cl_command_queue queue;
};

class Buffer
{
    public:
        Buffer(cl_mem buffer, size_t size);
        virtual ~Buffer();

        size_t getSize();

    private:
        cl_mem buffer;
        size_t size;

    friend Kernel;
    friend CommandQueue;
};

#endif // OPENCL_H
