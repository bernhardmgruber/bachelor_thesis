#ifndef OPENCL_H
#define OPENCL_H

// the target OpenCL version
#define OPENCL_VERSION 110 // OpenCL 1.1
//#define OPENCL_VERSION 120 // OpenCL 1.2

#if OPENCL_VERSION < 120
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#endif
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <string>
#include <vector>
#include <exception>

using namespace std;

class Context;
class CommandQueue;
class Program;
class Kernel;
class Buffer;
class Image;

/**
 * Exception class for OpenCL errors.
 */
class OpenCLException : public exception
{
    public:
        OpenCLException(string msg);

        virtual const char* what() throw();

    private:
        string msg;
};

/**
 * Checks the given error flag and raises an exception on error.
 *
 * @param error The OpenCL error flag to check.
 * @param name The name of the function in which checkError is called. Use the __LINE__ macro.
 * @throw Throws an OpenCLException if the
 */
void checkError(cl_int error, int line, string name) throw(OpenCLException);

/**
 * Singleton.
 */
class OpenCL
{
    public:
        /**
         * Initializes OpenCL on the first available platform.
         */
        static void init();

        /**
         * Cleans up all allocated resources.
         */
        static void cleanup();

        /**
         * Creates a GPU context on the first available platform.
         */
        static Context* getGPUContext();

        /**
         * Creates a CPU context on the first available platform.
         */
        static Context* getCPUContext();

    private:
        // Private constructor. No instance can be created from this class.
        OpenCL();
        virtual ~OpenCL();

        /** The platform id of the first available OpenCL platform. */
        static cl_platform_id platform;
};

/**
 * Represents a device and a context opened at this device.
 */
class Context
{
    public:
        /**
         * Constructor.
         * Creates a new context object.
         *
         * @param context The OpenCL context.
         * @param device The OpenCL device id.
         */
        Context(cl_context context, cl_device_id device);

        /**
         * Destructor.
         * Deletes the context object. All pending operations will be executed and all depending objects will still be valid.
         * The OpenCL Runtime deletes the object when there are no more references from other objects.
         */
        virtual ~Context();

        /**
         * Creates a new program from source.
         *
         * @param source The source code of the program.
         * @param options The command line options for the OpenCL compiler.
         * @return Returns a new instance of Program. This instance has to be deleted by the user.
         */
        Program* createProgram(string source, string options = "");

        /**
         * Creates a new command queue to this context.
         *
         * @return Returns a new instance of CommandQueue. This instance has to be deleted by the user.
         */
        CommandQueue* createCommandQueue();

        /**
         * Creates a new buffer.
         *
         * @param flags The OpenCL memory flags for this buffer.
         * @param size The size of the buffer in bytes.
         * @param ptr An optional pointer to an allocated memory from which memory can be copied.
         * @return Returns a new instance of Buffer. This instance has to be deleted by the user.
         */
        Buffer* createBuffer(cl_mem_flags flags, size_t size, void* ptr = nullptr);

        /**
         * Creates a new image.
         *
         * @param flags the OpenCL memory flags for this image/buffer.
         * @param format The OpenCL image format for this image.
         * @param descriptor The OpenCL image desccriptor for this image.
         * @param ptr An optional pointer to an allocated memory from which memory can be copied.
         * @return Returns a new instance of Image. This instance has to be deleted by the user.
         */
        Image* createImage(cl_mem_flags flags, const cl_image_format& format, const cl_image_desc& descriptor, void* ptr = nullptr);

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
            checkError(error, __LINE__, __FUNCTION__);

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

        /**
         * Gets the internal OpenCL device id.
         */
        cl_device_id getCLDevice();

        /**
         * Gets the internal OpenCL context.
         */
        cl_context getCLContext();

    private:
        /**
         * Reads a text file.
         *
         * @param The path to the file to read.
         * @return Returns the file's content as string.
         */
        string readFile(string fileName);

        /** The OpenCL device id. */
        cl_device_id device;

        /** The OpenCL context. */
        cl_context context;

    friend Kernel;
};

/**
 * Tempalte specialization for querying a information string from a device.
 *
 * @param info One of the predefined values for device information that usually returns a char*.
 */
template <>
string Context::getInfo<string>(cl_device_info info);

/**
 * Represents an OpenCL program.
 */
class Program
{
    public:
        /**
         * Constructor.
         * Creates a new program object.
         *
         * @param program The OpenCL program.
         * @param context The OpenCL context.
         */
        Program(cl_program program, Context* context);

        /**
         * Destructor.
         * Deletes the program object. All pending operations will be executed and all depending objects will still be valid.
         * The OpenCL Runtime deletes the object when there are no more references from other objects.
         */
        virtual ~Program();

        /**
         * Creates a new kernel from this program.
         *
         * @param entry The name of the __kernel function used as entry point.
         */
        Kernel* createKernel(string entry);

        /**
         * Gets the interal OpenCL program.
         */
        cl_program getCLProgram();

    private:
        /** The interal OpenCL program. */
        cl_program program;

        /// The context and device this program has been created for.
        Context* context;
};

/**
 * Represents an OpenCL kernel.
 */
class Kernel
{
    public:
        /**
         * Constructor.
         * Creates a new kernel object.
         *
         * @param kernel The OpenCL kernel.
         * @param context The OpenCL context.
         */
        Kernel(cl_kernel kernel, Context* context);

        /**
         * Destructor.
         * Deletes the kernel object. All pending operations will be executed and all depending objects will still be valid.
         * The OpenCL Runtime deletes the object when there are no more references from other objects.
         */
        ~Kernel();

        /**
         * Sets a buffer as argument to this kernel.
         *
         * @param index The index of the argument in the __kernel function declaration.
         * @param buffer The buffer to set as argument to this kernel.
         */
        void setArg(cl_uint index, Buffer* buffer);

        /**
         * Sets an image as argument to this kernel.
         *
         * @param index The index of the argument in the __kernel function declaration.
         * @param image The image to set as argument to this kernel.
         */
        void setArg(cl_uint index, Image* image);

        /**
         * Sets an argument to this kernel.
         *
         * @param index The index of the argument in the __kernel function declaration.
         * @param size The size of the argument in bytes.
         * @param value A pointer to size bytes of memory that are used as value for this argument.
         */
        void setArg(cl_uint index, size_t size, const void* value);

        /**
         * Sets an argument to this kernel.
         *
         * @param index The index of the argument in the __kernel function declaration.
         * @param value The value of this argument.
         * @param T The type of the value of this argument.
         */
        template <typename T>
        void setArg(cl_uint index, T value)
        {
            cl_int error = clSetKernelArg(kernel, index, sizeof(T), &value);
            checkError(error, __LINE__, __FUNCTION__);
        }

        /**
         * Queries the maximum work group size that can be used to execute this kernel.
         */
        size_t getWorkGroupSize();

        /**
         * Gets the internal OpenCL kernel.
         */
        cl_kernel getCLKernel();

    private:
        /** The internal OpenCL kernel. */
        cl_kernel kernel;

        /** The context and device this kernel has been created for. */
        Context* context;

    friend CommandQueue;
};

/**
 * Represents an OpenCL command queue.
 */
class CommandQueue
{
    public:
        /**
         * Constructor.
         * Creates a new command queue object.
         *
         * @param kernel The OpenCL command queue.
         * @param context The OpenCL context.
         */
        CommandQueue(cl_command_queue queue, Context* context);

        /**
         * Destructor.
         * Deletes the command queue object. All pending operations will be executed and all depending objects will still be valid.
         * The OpenCL Runtime deletes the object when there are no more references from other objects.
         */
        virtual ~CommandQueue();

        /**
         * Enqueues a kernel in this command queue.
         *
         * @param kernel The kernel to enqueue.
         * @param dimension The dimension of the kernel. This can be at least 1, 2 or 3.
         * @param globalWorkSizes A pointer to an array of dimension elements containing the size of a global work in each dimension.
         * @param localWorkSizes A pointer to an array of dimension elements containing the size of a work group in each dimension.
         * @param globalWorkOffsets A pointer to an array of dimension elements containing the offsets to the indexes retrievable via get_global_id() inside the kernels for each dimension.
         */
        void enqueueKernel(Kernel* kernel, cl_uint dimension, const size_t* globalWorkSizes, const size_t* localWorkSizes = nullptr, const size_t* globalWorkOffsets = nullptr);

        /**
         * Enqueues a kernel as a task in this command queue. The kernel will we executed only once.
         *
         * @param kernel The kernel to execute.
         */
        void enqueueTask(Kernel* kernel);

        /**
         * Enqueues a buffer reading operation in this command queue.
         * The size of the read memory in bytes is equal to the buffer size.
         *
         * @param buffer The buffer from which to read.
         * @param destination The destination where the read data should be written to.
         * @param blocking If set to true (default) the read operation blocks until it has finished.
         */
        void enqueueRead(Buffer* buffer, void* destination, bool blocking = true);

        /**
         * Enqueues a buffer reading operation in this command queue.
         *
         * @param buffer The buffer from which to read.
         * @param destination The destination where the read data should be written to.
         * @param offset The offset into the buffer from which to read.
         * @param size The size in bytes of the memory block inside the buffer that should be read.
         * @param blocking If set to true (default) the read operation blocks until it has finished.
         */
        void enqueueRead(Buffer* buffer, void* destination, size_t offset, size_t size, bool blocking = true);

        /**
         * Enqueues an image reading operation in this command queue.
         * The size of the read memory in bytes is equal to the image size.
         *
         * @param image The image from which to read.
         * @param destination The destination where the read data should be written to.
         * @param blocking If set to true (default) the read operation blocks until it has finished.
         */
        void enqueueRead(Image* image, void* destination, bool blocking = true);

        /**
         * Enqueues a buffer reading operation in this command queue.
         *
         * @param buffer The buffer where data is read from.
         * @param destination A pointer to memory where data is written to.
         * @param bufferOffset The offsets into the buffer where the data rectancle or data cube should be read from.
         * @param hostOffset The offsets into the host's memory where the data rectancle or data cube should be written to.
         * @param sizes The sizes of the data rectangle or data cube.
         * @param bufferRowLength The length in bytes of a row in the buffer. If set to zero this value defaults to size[0].
         * @param bufferSliceLength The length in bytes of a slice in the buffer. If set to zero this value defaults to size[1] * bufferRowLength.
         * @param hostRowLength The length in bytes of a row in the host's memory. If set to zero this value defaults to size[0].
         * @param hostSliceLength The length in bytes of a slice in the host's memory. If set to zero this value defaults to size[1] * bufferRowLength.
         * @param blocking If set to true (default) the read operation blocks until it has finished.
         */
        void enqueueReadRect(Buffer* buffer, void* destination, const size_t bufferOffset[3], const size_t hostOffset[3], const size_t sizes[3], size_t bufferRowLength, size_t bufferSliceLength, size_t hostRowLength, size_t hostSliceLength, bool blocking = true);

        /**
         * Enqueues a buffer writing operation in this command queue.
         * The size of the written memory in bytes is equal to the buffer size.
         *
         * @param buffer The buffer where data is written to.
         * @param source A pointer to memory where data is read from.
         * @param blocking If set to true (default) the write operation blocks until it has finished.
         */
        void enqueueWrite(Buffer* buffer, const void* source, bool blocking = true);

        /**
         * Enqueues a buffer writing operation in this command queue.
         *
         * @param buffer The buffer where data is written to.
         * @param source A pointer to memory where data is read from.
         * @param offset The offset into the buffer where the data should be written.
         * @param size The size in bytes of the memory block inside the buffer that should be written.
         * @param blocking If set to true (default) the write operation blocks until it has finished.
         */
        void enqueueWrite(Buffer* buffer, const void* source, size_t offset, size_t size, bool blocking = true);

        /**
         * Enqueues a buffer writing operation in this command queue.
         *
         * @param buffer The buffer where data is written to.
         * @param source A pointer to memory where data is read from.
         * @param bufferOffset The offsets into the buffer where the data rectancle or data cube should be written.
         * @param hostOffset The offsets into the host's memory where the data rectancle or data cube should be read from.
         * @param sizes The sizes of the data rectangle or data cube.
         * @param bufferRowLength The length in bytes of a row in the buffer. If set to zero this value defaults to size[0].
         * @param bufferSliceLength The length in bytes of a slice in the buffer. If set to zero this value defaults to size[1] * bufferRowLength.
         * @param hostRowLength The length in bytes of a row in the host's memory. If set to zero this value defaults to size[0].
         * @param hostSliceLength The length in bytes of a slice in the host's memory. If set to zero this value defaults to size[1] * bufferRowLength.
         * @param blocking If set to true (default) the write operation blocks until it has finished.
         */
        void enqueueWriteRect(Buffer* buffer, const void* source, const size_t bufferOffset[3], const size_t hostOffset[3], const size_t sizes[3], size_t bufferRowLength, size_t bufferSliceLength, size_t hostRowLength, size_t hostSliceLength, bool blocking = true);

        /**
         * Enqueues a image writing operation in this command queue.
         * The size of the written memory in bytes is equal to the image size.
         *
         * @param image The image where data is written to.
         * @param source A pointer to memory where data is read from.
         * @param blocking If set to true (default) the write operation blocks until it has finished.
         */
        void enqueueWrite(Image* image, const void* source, bool blocking = true);

        /**
         * Enqueues an image map operation in this command queue.
         *
         * @param image The image to map.
         * @param flags The OpenCL memory flags for this map operation.
         * @param blocking If set to true (default) the map operation blocks until it has finished.
         * @return Returns a pointer to the mapped memory location of the image.
         */
        void* enqueueMap(Image* image, cl_map_flags flags, bool blocking = true);

        /**
         * Enqueues an image unmap operation in this command queue.
         *
         * @param image The image to unmap.
         * @param ptr The pointer returned by enqueueMap().
         */
        void enqueueUnmap(Image* image, void* ptr);

        /**
         * Enqueues a buffer copy operation in this command queue.
         * The number of bytes copied is equal to the size of the destination buffer.
         *
         * @param src The buffer to copy from.
         * @param dest The buffer to copy to.
         */
        void enqueueCopy(Buffer* src, Buffer* dest);

        /**
         * Enqueues a buffer copy operation in this command queue.
         *
         * @param src The buffer to copy from.
         * @param dest The buffer to copy to.
         * @param srcOffset The offset into the source buffer where the data is read from.
         * @param destOffset The offset into the destination buffer where the data is written to.
         * @param size The number of bytes to copy.
         */
        void enqueueCopy(Buffer* src, Buffer* dest, size_t srcOffset, size_t destOffset, size_t size);

        /**
         * Enqueues a fill buffer operation.
         *
         * @param buffer The buffer to fill.
         * @param val The value to fill the buffer with.
         */
        template <typename T>
        void enqueueFill(Buffer* buffer, T val);

        /**
         * Enqueues a barrier operation in this command queue.
         */
        void enqueueBarrier();

        /**
         * Issues all enqueued operations to the device.
         */
        void flush();

        /**
         * Waits for all enqueued operations to finish.
         */
        void finish();

        /**
         * Gets the context for this command queue.
         */
        Context* getContext();

        /**
         * Gets the internal OpenCL command queue.
         */
        cl_command_queue getCLCommandQueue();

    private:
        /** The internal OpenCL command queue. */
        cl_command_queue queue;

        /** The context and device this kernel has been created for. */
        Context* context;
};

/**
 * Represents a buffer.
 */
class Buffer
{
    public:
        /**
         * Constructor.
         * Creates a new buffer object.
         *
         * @param buffer The OpenCL buffer.
         * @param size The size of the buffer in bytes
         */
        Buffer(cl_mem buffer, size_t size);

        /**
         * Destructor.
         * Deletes the buffer object. All pending operations will be executed and all depending objects will still be valid.
         * The OpenCL Runtime deletes the object when there are no more references from other objects.
         */
        virtual ~Buffer();

        /**
         * Returns the size of the buffer.
         */
        size_t getSize();

        /**
         * Gets the internal OpenCL buffer.
         */
        cl_mem getCLBuffer();

    private:
        /** The internal OpenCL buffer. */
        cl_mem buffer;

        /** The size of the buffer. */
        size_t size;

    friend Kernel;
    friend CommandQueue;
};

/**
 * Represents an image.
 */
class Image
{
    public:
        /**
         * Constructor.
         * Creates a new image object.
         *
         * @param buffer The OpenCL buffer/image.
         * @param format The OpenCL image format for this image.
         * @param descriptor The OpenCL image desccriptor for this image.
         */
        Image(cl_mem buffer, const cl_image_format& format, const cl_image_desc& descriptor);

        /**
         * Destructor.
         * Deletes the image object. All pending operations will be executed and all depending objects will still be valid.
         * The OpenCL Runtime deletes the object when there are no more references from other objects.
         */
        virtual ~Image();

        /**
         * Gets the internal OpenCL image format.
         */
        cl_image_format getFormat();

        /**
         * Gets the internal OpenCL image descriptor.
         */
        cl_image_desc getDescriptor();

        /**
         * Gets the internal OpenCL image/buffer.
         */
        cl_mem getCLBuffer();

    private:
        /** The internal OpenCL image/buffer. */
        cl_mem buffer;

        /** The internal OpenCL image format. */
        const cl_image_format format;

        /** The internal OpenCL image descriptor. */
        const cl_image_desc descriptor;

    friend Kernel;
    friend CommandQueue;
};

//
// TEMPLATE METHODS
//

template <typename T>
void CommandQueue::enqueueFill(Buffer* buffer, T val)
{
    cl_int error = clEnqueueFillBuffer(queue, buffer->buffer, &val, sizeof(T), 0, buffer->size, 0, nullptr, nullptr);
    checkError(error, __LINE__, __FUNCTION__);
}

#endif // OPENCL_H
