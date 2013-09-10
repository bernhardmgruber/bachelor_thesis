#include <CL/cl.h>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <exception>
#include <stdexcept>

using namespace std;

int roundToMultiple(int x, int multiple)
{
	if(x % multiple == 0)
		return x;
	else
		return (x / multiple + 1) * multiple;
}

void multNaiveCPU(float* a, float* b, float* c, size_t n) {
	for (size_t row = 0; row < n; row++) {
		for (size_t col = 0; col < n; col++) {
			c[row * n + col] = 0;
			for (size_t i = 0; i < n; i++)
				c[row * n + col] += a[row * n + i] * b[i * n + col];
		} // for
	} // for
} // multNaiveCPU

void multNaiveGPU(float* a, float* b, float* c, cl_uint n,
		cl_context context, cl_command_queue queue, cl_kernel kernel, size_t workGroupSize) {
	size_t bufferSize = n * n * sizeof(float);

	cl_mem aBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, nullptr, nullptr);
	cl_mem bBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, nullptr, nullptr);
	cl_mem cBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, nullptr);

	clEnqueueWriteBuffer(queue, aBuffer, false, 0, bufferSize, a, 0, nullptr, nullptr);
	clEnqueueWriteBuffer(queue, bBuffer, false, 0, bufferSize, b, 0, nullptr, nullptr);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuffer);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &bBuffer);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &cBuffer);
	clSetKernelArg(kernel, 3, sizeof(cl_uint), &n);
	size_t adjustedWS = roundToMultiple(n, workGroupSize);
	size_t globalWS[] = { adjustedWS, adjustedWS };
	size_t localWS[] = { workGroupSize, workGroupSize };
	clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWS, localWS, 0, nullptr, nullptr);

	clEnqueueReadBuffer(queue, cBuffer, true, 0, bufferSize, c, 0, nullptr, nullptr);

	clReleaseMemObject(aBuffer);
	clReleaseMemObject(bBuffer);
	clReleaseMemObject(cBuffer);
} // multNaiveGPU

#define TILE_SIZE 16

void multTilesGPU(float* a, float* b, float* c, cl_uint n,
		cl_context context, cl_command_queue queue, cl_kernel kernel) {
	cl_uint adjustedSize = roundToMultiple(n, TILE_SIZE);
	size_t bufferSize = adjustedSize * adjustedSize * sizeof(float);
	cl_mem aBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, nullptr, nullptr);
	cl_mem bBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, nullptr, nullptr);
	cl_mem cBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, nullptr);

	size_t bufferOffset[] = {0, 0, 0};
	size_t hostOffset[]   = {0, 0, 0};
	size_t sizes[]        = {n * sizeof(float), n, 1};

	if (adjustedSize != n) {
		float zero = 0.0f;
		clEnqueueFillBuffer(queue, aBuffer, &zero, sizeof(float), 0, bufferSize, 0, nullptr, nullptr);
		clEnqueueFillBuffer(queue, bBuffer, &zero, sizeof(float), 0, bufferSize, 0, nullptr, nullptr);
		clEnqueueWriteBufferRect(queue, aBuffer, false, bufferOffset, hostOffset, sizes,
				adjustedSize * sizeof(float), 0, n * sizeof(float), 0, a, 0, nullptr, nullptr);
		clEnqueueWriteBufferRect(queue, bBuffer, false, bufferOffset, hostOffset, sizes,
				adjustedSize * sizeof(float), 0, n * sizeof(float), 0, b, 0, nullptr, nullptr);
	} else {
		clEnqueueWriteBuffer(queue, aBuffer, false, 0, bufferSize, a, 0, nullptr, nullptr);
		clEnqueueWriteBuffer(queue, bBuffer, false, 0, bufferSize, b, 0, nullptr, nullptr);
	} // if

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuffer);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &bBuffer);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &cBuffer);
	clSetKernelArg(kernel, 3, sizeof(cl_uint), &adjustedSize);
	size_t globalWS[] = { adjustedSize, adjustedSize };
	size_t localWS[] = { TILE_SIZE, TILE_SIZE };
	clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWS, localWS, 0, nullptr, nullptr);

	if (adjustedSize != n)
		clEnqueueReadBufferRect(queue, cBuffer, true, bufferOffset, hostOffset, sizes,
				adjustedSize * sizeof(float), 0, n * sizeof(float), 0, c, 0, nullptr, nullptr);
	else
		clEnqueueReadBuffer(queue, cBuffer, true, 0, bufferSize, c, 0, nullptr, nullptr);

	clReleaseMemObject(aBuffer);
	clReleaseMemObject(bBuffer);
	clReleaseMemObject(cBuffer);
} // multTilesGPU

#define BLOCK_SIZE 4

void matrixMulCLRect(float* a, float* b, float* c, cl_uint n, cl_context context, cl_command_queue queue, cl_kernel kernel, size_t workGroupSize, int multiple, int elementsPerThread) {
	cl_uint adjustedSize = roundToMultiple(n, multiple);
	size_t bufferSize = adjustedSize * adjustedSize * sizeof(float);

	cl_mem aBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, nullptr, nullptr);
	cl_mem bBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, nullptr, nullptr);
	cl_mem cBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, nullptr);

	size_t bufferOffset[] = {0, 0, 0};
	size_t hostOffset[] = {0, 0, 0};
	size_t sizes[] = {n * sizeof(float), n, 1};

	if(adjustedSize != n) {
		float zero = 0.0f;
		clEnqueueFillBuffer(queue, aBuffer, &zero, sizeof(float), 0, bufferSize, 0, nullptr, nullptr);
		clEnqueueFillBuffer(queue, bBuffer, &zero, sizeof(float), 0, bufferSize, 0, nullptr, nullptr);

		clEnqueueWriteBufferRect(queue, aBuffer, false, bufferOffset, hostOffset, sizes, adjustedSize * sizeof(float), 0, n * sizeof(float), 0, a, 0, nullptr, nullptr);
		clEnqueueWriteBufferRect(queue, bBuffer, false, bufferOffset, hostOffset, sizes, adjustedSize * sizeof(float), 0, n * sizeof(float), 0, b, 0, nullptr, nullptr);
	} else {
		clEnqueueWriteBuffer(queue, aBuffer, false, 0, bufferSize, a, 0, nullptr, nullptr);
		clEnqueueWriteBuffer(queue, bBuffer, false, 0, bufferSize, b, 0, nullptr, nullptr);
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuffer);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &bBuffer);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &cBuffer);
	clSetKernelArg(kernel, 3, sizeof(cl_uint), &adjustedSize);

	size_t adjustedWS = roundToMultiple(adjustedSize, workGroupSize * elementsPerThread);
	size_t globalWS[] = { adjustedWS / elementsPerThread, adjustedWS / elementsPerThread };
	size_t localWS[] = { workGroupSize, workGroupSize };
	clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWS, localWS, 0, nullptr, nullptr);

	if(adjustedSize != n)
		clEnqueueReadBufferRect(queue, cBuffer, true, bufferOffset, hostOffset, sizes, adjustedSize * sizeof(float), 0, n * sizeof(float), 0, c, 0, nullptr, nullptr);
	else
		clEnqueueReadBuffer(queue, cBuffer, true, 0, bufferSize, c, 0, nullptr, nullptr);

	clReleaseMemObject(aBuffer);
	clReleaseMemObject(bBuffer);
	clReleaseMemObject(cBuffer);
}

#if 0
#define BLOCK_SIZE 4
	...
	cl_uint adjustedSize = roundToMultiple(n, BLOCK_SIZE);
	...
	size_t adjustedWS = roundToMultiple(adjustedSize, workGroupSize * BLOCK_SIZE);
	size_t globalWS[] = { adjustedWS / BLOCK_SIZE, adjustedWS / BLOCK_SIZE };
	size_t localWS[] = { workGroupSize, workGroupSize };
	...
#endif

#if 0
#define TILE_SIZE 16
#define BLOCK_SIZE 4
	...
	cl_uint adjustedSize = roundToMultiple(n, BLOCK_SIZE * TILE_SIZE);
	...
	size_t globalWS[] = { adjustedSize / BLOCK_SIZE, adjustedSize / BLOCK_SIZE };
	...
#endif


string readFile(string fileName)
{
	ifstream file(fileName, ios::in);
	if(!file)
		throw runtime_error("Error opening file " + fileName);

	string buffer = string((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());

	file.close();

	return buffer;
}

bool memcmp_float(const float* a, const float* b, size_t count, float epsilon = 0.001)
{
	for(size_t i = 0; i < count; i++)
		if(fabs(a[i] - b[i]) > epsilon)
			return false;

	return true;
}

int main(int argc, char* argv[])
{
	// get the first available platform
	cl_platform_id platform;
	clGetPlatformIDs(1, &platform, nullptr);

	// get the first available GPU on this platform
	cl_device_id device;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

	// create context
	cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);

	// create command queue
	cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);

	// create OpenCL program from source code
	string source1 = readFile("../../../src/matrix/gpu/thesis/Mult.cl");
	string source2 = readFile("../../../src/matrix/gpu/thesis/MultLocal.cl");
	string source3 = readFile("../../../src/matrix/gpu/thesis/MultBlock.cl");
	string source4 = readFile("../../../src/matrix/gpu/thesis/MultBlockLocal.cl");

	const char* source1Ptr = source1.c_str();
	const char* source2Ptr = source2.c_str();
	const char* source3Ptr = source3.c_str();
	const char* source4Ptr = source4.c_str();

	cl_program program1 = clCreateProgramWithSource(context, 1, &source1Ptr, nullptr, nullptr);
	cl_program program2 = clCreateProgramWithSource(context, 1, &source2Ptr, nullptr, nullptr);
	cl_program program3 = clCreateProgramWithSource(context, 1, &source3Ptr, nullptr, nullptr);
	cl_program program4 = clCreateProgramWithSource(context, 1, &source4Ptr, nullptr, nullptr);

	// compile the program for the device
	clBuildProgram(program1, 1, &device, "", nullptr, nullptr);
	clBuildProgram(program2, 1, &device, "", nullptr, nullptr);
	clBuildProgram(program3, 1, &device, "", nullptr, nullptr);
	clBuildProgram(program4, 1, &device, "", nullptr, nullptr);

	// create the kernel
	cl_kernel kernel1 = clCreateKernel(program1, "MultNaive", nullptr);
	cl_kernel kernel2 = clCreateKernel(program2, "MultTiles", nullptr);
	cl_kernel kernel3 = clCreateKernel(program3, "MultBlocks", nullptr);
	cl_kernel kernel4 = clCreateKernel(program4, "MultBlocksAndTiles", nullptr);

	// MATRIX MUL
	int n = 1000;
	float* a = new float[n * n];
	float* b = new float[n * n];
	float* c0 = new float[n * n]();
	float* c1 = new float[n * n]();
	float* c2 = new float[n * n]();
	float* c3 = new float[n * n]();
	float* c4 = new float[n * n]();

	generate(a, a + n * n, []()
	{
		return rand() % 100;
	});
	generate(b, b + n * n, []()
	{
		return rand() % 100;
	});

	multNaiveCPU(a, b, c0, n);

	multNaiveGPU(a, b, c1, n, context, queue, kernel1, 16); // 2D
	if(!memcmp_float(c0, c1, n * n))
		cerr << "validation of kernel 1 failed" << endl;
	else
		cout << "kernel 1 ok" << endl;

	multTilesGPU(a, b, c2, n, context, queue, kernel2); // 2D
	if(!memcmp_float(c0, c2, n * n))
		cerr << "validation of kernel 2 failed" << endl;
	else
		cout << "kernel 2 ok" << endl;

	matrixMulCLRect(a, b, c3, n, context, queue, kernel3, 16, BLOCK_SIZE, BLOCK_SIZE); // 2D
	if(!memcmp_float(c0, c3, n * n))
		cerr << "validation of kernel 3 failed" << endl;
	else
		cout << "kernel 3 ok" << endl;

	matrixMulCLRect(a, b, c4, n, context, queue, kernel4, 16, BLOCK_SIZE * TILE_SIZE, BLOCK_SIZE); // 2D
	if(!memcmp_float(c0, c4, n * n))
		cerr << "validation of kernel 4 failed" << endl;
	else
		cout << "kernel 4 ok" << endl;

	delete[] a;
	delete[] b;
	delete[] c0;
	delete[] c1;
	delete[] c2;
	delete[] c3;
	delete[] c4;

	clReleaseKernel(kernel1);
	clReleaseKernel(kernel2);
	clReleaseKernel(kernel3);
	clReleaseKernel(kernel4);
	clReleaseProgram(program1);
	clReleaseProgram(program2);
	clReleaseProgram(program3);
	clReleaseProgram(program4);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	clReleaseDevice(device);

	return 0;
}
