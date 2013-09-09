#include <CL/cl.h>
#include <iostream>

int main(int argc, char* argv[]) {
	cl_int error;

	cl_platform_id platform;
	error = clGetPlatformIDs(1, &platform, nullptr);

	cl_device_id device;
	error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

	cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
	cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error);

	const char* source = ""
	"__kernel void vectorAdd(__global float* a, __global float* b, __global float* c)"
	"{"
	"	size_t id = get_global_id(0);"
	"	c[id] = a[id] + b[id];"
	"}";
	cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &error);
	error = clBuildProgram(program, 1, &device, "", nullptr, nullptr);

	cl_kernel kernel = clCreateKernel(program, "vectorAdd", &error);

	const size_t N = 1024;
	float* a = new float[N];
	float* b = new float[N];
	float* c = new float[N];

	for (size_t i = 0; i < N; i++) {
		a[i] = i;
		b[i] = N - i;
	} // for

	cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY,  N * sizeof(float), nullptr, &error);
	cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY,  N * sizeof(float), nullptr, &error);
	cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float), nullptr, &error);

	error = clEnqueueWriteBuffer(queue, bufferA, false, 0, N * sizeof(float), a, 0, nullptr, nullptr);
	error = clEnqueueWriteBuffer(queue, bufferB, false, 0, N * sizeof(float), b, 0, nullptr, nullptr);

	error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
	error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
	error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
	error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &N, nullptr, 0, nullptr, nullptr);

	error = clEnqueueReadBuffer(queue, bufferC, true, 0, N * sizeof(float), c, 0, nullptr, nullptr);

	for (size_t i = 0; i < N; i++)
		std::cout << c[i] << ", ";

	delete[] a;
	delete[] b;
	delete[] c;

	error = clReleaseMemObject(bufferA);
	error = clReleaseMemObject(bufferB);
	error = clReleaseMemObject(bufferC);
	error = clReleaseKernel(kernel);
	error = clReleaseProgram(program);
	error = clReleaseCommandQueue(queue);
	error = clReleaseContext(context);
	error = clReleaseDevice(device);

	return 0;
} // main
