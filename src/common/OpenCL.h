#ifndef OPENCL_H
#define OPENCL_H

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <string>
#include <vector>
#include <exception>

using namespace std;

void checkError(cl_int error, int line);

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

        /**
         * Retrieves an OpenCL device information.
         *
         * @param info An enumeration constant that identifies the device information being queried. @see http://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clGetDeviceInfo.html
         * @throw OpenCLException If the info parameter is invalid or querying the requested value failed.
         */
        template <typename T>
        T getInfo(cl_device_info info)
        {
            T value = 0;
            cl_int error = clGetDeviceInfo(device, info, sizeof(T), (void*) &value, nullptr);
            checkError(error, __LINE__);

            return value;
        }

        /**
         * Same as getInfo<T>() but returns the template argument's default value on error instread of throwing an exception.
         */
        template <typename T>
        T getInfoWithDefaultOnError(cl_device_info info)
        {
            try
            {
                return getInfo<T>(info);
            }
            catch(...)
            {
                return T();
            }
        }

        /**
         * Gets information about the requested property. The required size of the value is determined and allocated by this function. The user is responsible for freeing the returned buffer using the delete operator.
         */
        void* getInfo(cl_device_info);

        /**
         * Same as getInfo() but returns nullptr on error instread of throwing an exception.
         */
        void* getInfoWithDefaultOnError(cl_device_info);

        cl_device_id getCLDevice();
        cl_context getCLContext();

    private:
        string readFile(string fileName);

        cl_device_id device;
        cl_context context;

        //vector<Program*> programs;
        //vector<CommandQueue*> queues;
        //vector<Buffer*> buffers;
        friend Kernel;
};

/**
 * Tempalte specialization for querying a information string from a device.
 *
 * @param info One of the predefined values for device information that usually returns a char*.
 */
template <>
string Context::getInfo<string>(cl_device_info info);

class Program
{
    public:
        Program(cl_program program, Context* context);
        virtual ~Program();

        Kernel* createKernel(string entry);

        cl_program getCLProgram();

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
            cl_int error = clSetKernelArg(kernel, index, sizeof(T), &value);
            checkError(error, __LINE__);
        }

        size_t getWorkGroupSize();

        cl_kernel getCLKernel();

    private:
        cl_kernel kernel;

        /// The context and device this kernel has been created for.
        Context* context;

    friend CommandQueue;
};

class CommandQueue
{
    public:
        CommandQueue(cl_command_queue queue, Context* context);
        virtual ~CommandQueue();

        void enqueueKernel(Kernel* kernel, cl_uint dimension, const size_t* globalWorkSizes, const size_t* localWorkSizes = nullptr, const size_t* globalWorkOffsets = nullptr);
        void enqueueTask(Kernel* kernel);
        void enqueueRead(Buffer* buffer, void* destination, bool blocking = true);
        void enqueueRead(Buffer* buffer, void* destination, size_t offset, size_t size, bool blocking = true);
        void enqueueWrite(Buffer* buffer, const void* source, bool blocking = true);
        void enqueueWrite(Buffer* buffer, const void* source, size_t offset, size_t size, bool blocking = true);
        void enqueueCopy(Buffer* src, Buffer* dest);
        void enqueueCopy(Buffer* src, Buffer* dest, size_t srcOffset, size_t destOffset, size_t size);
        void enqueueBarrier();

        void flush();
        void finish();

        Context* getContext();

        cl_command_queue getCLCommandQueue();

    private:
        cl_command_queue queue;

        /// The context and device this kernel has been created for.
        Context* context;
};

class Buffer
{
    public:
        Buffer(cl_mem buffer, size_t size);
        virtual ~Buffer();

        size_t getSize();

        cl_mem getCLBuffer();

    private:
        cl_mem buffer;
        size_t size;

    friend Kernel;
    friend CommandQueue;
};

#endif // OPENCL_H
