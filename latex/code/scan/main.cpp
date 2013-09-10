#include <CL/cl.h>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <exception>
#include <stdexcept>
#include <stdint.h>

#define int cl_int

using namespace std;

size_t roundToMultiple(size_t x, size_t multiple)
{
	if(x % multiple == 0)
		return x;
	else
		return (x / multiple + 1) * multiple;
}

cl_uint roundToPowerOfTwo(cl_uint x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;

	return x + 1;
}

void scanCPU(int* data, int* result, size_t n) {
	result[0] = 0;
	for (size_t i = 1; i < n; i++)
		result[i] = result[i - 1] + data[i - 1];
} // scanCPU

void scanNaiveGPU(int* data, int* result, cl_uint n,
		cl_context context, cl_command_queue queue, cl_kernel kernel, size_t workGroupSize) {
	size_t bufferSize = n * sizeof(int);
	cl_mem src = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize, nullptr, nullptr);
	cl_mem dst = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize, nullptr, nullptr);

	clEnqueueWriteBuffer(queue, src, false, 0, bufferSize, data, 0, nullptr, nullptr);

	for (cl_uint offset = 1; offset < n; offset <<= 1) {
		clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst);
		clSetKernelArg(kernel, 2, sizeof(cl_uint), &offset);
		clSetKernelArg(kernel, 3, sizeof(cl_uint), &n);
		size_t globalWS[] = { roundToMultiple(n, workGroupSize) };
		size_t localWS[]  = { workGroupSize };
		clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, globalWS, localWS, 0, nullptr, nullptr);

		std::swap(src, dst);
	} // for

	clEnqueueReadBuffer(queue, src, true, 0, bufferSize, result, 0, nullptr, nullptr);

	clReleaseMemObject(src);
	clReleaseMemObject(dst);
} // scanNaiveGPU

void scanWorkEfficientGPU(int* data, int* result, cl_uint n,
		cl_context context, cl_command_queue queue, cl_kernel upSweep, cl_kernel downSweep,
		size_t workGroupSize) {
	size_t adjustedSize = roundToPowerOfTwo(n);
	cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
			adjustedSize * sizeof(int), nullptr, nullptr);

	clEnqueueWriteBuffer(queue, buffer, false, 0, n * sizeof(int), data, 0, nullptr, nullptr);

	// upsweep
	size_t nodes = adjustedSize >> 1;
	for (cl_uint offset = 1; offset < adjustedSize; offset <<= 1, nodes >>= 1) {
		clSetKernelArg(upSweep, 0, sizeof(cl_mem), &buffer);
		clSetKernelArg(upSweep, 1, sizeof(cl_uint), &offset);
		size_t globalWS[] = { nodes };
		size_t localWS[] = { std::min(workGroupSize, nodes) };
		clEnqueueNDRangeKernel(queue, upSweep, 1, nullptr, globalWS, localWS, 0, nullptr, nullptr);
	} // for

	// set last element to zero
	cl_uint zero = 0;
	clEnqueueWriteBuffer(queue, buffer, false, (adjustedSize - 1) * sizeof(cl_uint), sizeof(cl_uint),
			&zero, 0, nullptr, nullptr);

	// downsweep
	nodes = 1;
	for (cl_uint offset = adjustedSize >> 1; offset >= 1; offset >>= 1, nodes <<= 1) {
		clSetKernelArg(downSweep, 0, sizeof(cl_mem), &buffer);
		clSetKernelArg(downSweep, 1, sizeof(cl_uint), &offset);
		size_t globalWS[] = { nodes };
		size_t localWS[] = { std::min(workGroupSize, nodes) };
		clEnqueueNDRangeKernel(queue, downSweep, 1, nullptr, globalWS, localWS, 0, nullptr, nullptr);
	} // for

	clEnqueueReadBuffer(queue, buffer, true, 0, n * sizeof(int), result, 0, nullptr, nullptr);

	clReleaseMemObject(buffer);
} // scanWorkEfficientGPU

void scanRecursiveGPU_r(cl_mem values, cl_uint n,
		cl_context context, cl_command_queue queue, cl_kernel scanBlocks, cl_kernel addSums,
		size_t workGroupSize) {
	size_t sumCount = roundToMultiple(n / (workGroupSize * 2), workGroupSize * 2);
	cl_mem sums = clCreateBuffer(context, CL_MEM_READ_WRITE, sumCount * sizeof(int), nullptr, nullptr);

	clSetKernelArg(scanBlocks, 0, sizeof(cl_mem), &values);
	clSetKernelArg(scanBlocks, 1, sizeof(cl_mem), &sums);
	clSetKernelArg(scanBlocks, 2, sizeof(int) * 2 * workGroupSize, nullptr);
	size_t globalWS[] = { n / 2 };
	size_t localWS[] = { workGroupSize };
	clEnqueueNDRangeKernel(queue, scanBlocks, 1, nullptr, globalWS, localWS, 0, nullptr, nullptr);

	if (n > workGroupSize * 2) {
		scanRecursiveGPU_r(sums, sumCount, context, queue, scanBlocks, addSums, workGroupSize);

		clSetKernelArg(addSums, 0, sizeof(cl_mem), &values);
		clSetKernelArg(addSums, 1, sizeof(cl_mem), &sums);
		clEnqueueNDRangeKernel(queue, addSums, 1, nullptr, globalWS, localWS, 0, nullptr, nullptr);
	} // if

	clReleaseMemObject(sums);
} // scanRecursiveGPU_r

void scanRecursiveGPU(int* data, int* result, cl_uint n,
		cl_context context, cl_command_queue queue, cl_kernel scanBlocks, cl_kernel addSums,
		size_t workGroupSize) {
	size_t adjustedSize = roundToMultiple(n, workGroupSize * 2);
	cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, adjustedSize * sizeof(int), nullptr, nullptr);

	clEnqueueWriteBuffer(queue, buffer, false, 0, n * sizeof(int), data, 0, nullptr, nullptr);
	scanRecursiveGPU_r(buffer, adjustedSize, context, queue, scanBlocks, addSums, workGroupSize);
	clEnqueueReadBuffer(queue, buffer, true, 0, n * sizeof(int), result, 0, nullptr, nullptr);

	clReleaseMemObject(buffer);
} // scanRecursiveGPU

#if 0
#define VECTOR_WIDTH 8
...
	size_t sumCount = roundToMultiple(n / (workGroupSize * 2 * VECTOR_WIDTH),
			workGroupSize * 2 * VECTOR_WIDTH);
	...
	size_t globalWS[] = { n / (2 * VECTOR_WIDTH) };
	...
	if (n > workGroupSize * 2 * VECTOR_WIDTH) {
		scanCLRecursiveVector_r(sums, sumCount, context, queue, scanBlocks, addSums, workGroupSize);
		...
#endif // 0

#if 0
#define CONCAT(a, b) a ## b           // concat token a and b
#define CONCAT_EXP(a, b) CONCAT(a, b) // concat token a and b AFTER expansion

#define UPSWEEP_STEP(left, right) right += left

#define UPSWEEP_STEPS(left, right) \
	UPSWEEP_STEP(CONCAT_EXPANDED(val1.s, left), CONCAT_EXPANDED(val1.s, right)); \
	UPSWEEP_STEP(CONCAT_EXPANDED(val2.s, left), CONCAT_EXPANDED(val2.s, right))

#define DOWNSWEEP_STEP_TMP(left, right, tmp) \
	int tmp = left; \
	left = right; \
	right += tmp

#define DOWNSWEEP_STEP(left, right) \
	DOWNSWEEP_STEP_TMP(left, right, CONCAT_EXPANDED(tmp, __COUNTER__))

#define DOWNSWEEP_STEPS(left, right) \
	DOWNSWEEP_STEP(CONCAT_EXPANDED(val1.s, left), CONCAT_EXPANDED(val1.s, right)); \
	DOWNSWEEP_STEP(CONCAT_EXPANDED(val2.s, left), CONCAT_EXPANDED(val2.s, right))

__kernel void ScanBlocksVec(__global int8* buffer, __global int* sums, __local int* shared) {
	uint globalId = get_global_id(0);
	uint localId  = get_local_id(0);
	uint n        = get_local_size(0) * 2;

	uint offset = 1;

	int8 val1 = buffer[2 * globalId + 0];
	int8 val2 = buffer[2 * globalId + 1];

	// upsweep vectors
	UPSWEEP_STEPS(0, 1);
	UPSWEEP_STEPS(2, 3);
	UPSWEEP_STEPS(4, 5);
	UPSWEEP_STEPS(6, 7);

	UPSWEEP_STEPS(1, 3);
	UPSWEEP_STEPS(5, 7);

	UPSWEEP_STEPS(3, 7);

	// move sums into shared memory block and clear last elements
	shared[2 * localId + 0] = val1.s7;
	shared[2 * localId + 1] = val2.s7;

	val1.s7 = 0;
	val2.s7 = 0;

	// downsweep vectors
	DOWNSWEEP_STEPS(3, 7);

	DOWNSWEEP_STEPS(1, 3);
	DOWNSWEEP_STEPS(5, 7);

	DOWNSWEEP_STEPS(0, 1);
	DOWNSWEEP_STEPS(2, 3);
	DOWNSWEEP_STEPS(4, 5);
	DOWNSWEEP_STEPS(6, 7);

	// scan block in local memory
	...

	// apply the sums
	val1 += shared[2 * localId + 0];
	val2 += shared[2 * localId + 1];

	buffer[2 * globalId + 0] = val1;
	buffer[2 * globalId + 1] = val2;
} // ScanBlocksVec

__kernel void AddSumsVec(__global int8* buffer, __global int* sums) {
	uint globalId = get_global_id(0);

	int val = sums[get_group_id(0)];

	buffer[globalId * 2 + 0] += val;
	buffer[globalId * 2 + 1] += val;
} // AddSumsVec
#endif // 0

#define VECTOR_WIDTH 8

void scanRecursiveVecGPU_r(cl_mem values, cl_uint n, cl_context context, cl_command_queue queue, cl_kernel scanBlocks, cl_kernel addSums, size_t workGroupSize)
{
	size_t sumBufferSize = roundToMultiple(n / (workGroupSize * 2 * VECTOR_WIDTH), workGroupSize * 2 * VECTOR_WIDTH);

	cl_mem sums = clCreateBuffer(context, CL_MEM_READ_WRITE, sumBufferSize * sizeof(int), nullptr, nullptr);

	clSetKernelArg(scanBlocks, 0, sizeof(cl_mem), &values);
	clSetKernelArg(scanBlocks, 1, sizeof(cl_mem), &sums);
	clSetKernelArg(scanBlocks, 2, sizeof(int) * 2 * workGroupSize, nullptr);

	size_t globalWS[] = { n / (2 * VECTOR_WIDTH) };
	size_t localWS[] = { workGroupSize };

	clEnqueueNDRangeKernel(queue, scanBlocks, 1, nullptr, globalWS, localWS, 0, nullptr, nullptr);

	if(n > workGroupSize * 2 * VECTOR_WIDTH)
	{
		scanRecursiveVecGPU_r(sums, sumBufferSize, context, queue, scanBlocks, addSums, workGroupSize);

		clSetKernelArg(addSums, 0, sizeof(cl_mem), &values);
		clSetKernelArg(addSums, 1, sizeof(cl_mem), &sums);

		clEnqueueNDRangeKernel(queue, addSums, 1, nullptr, globalWS, localWS, 0, nullptr, nullptr);
	}

	clReleaseMemObject(sums);
} // scanRecursiveVecGPU_r

void scanRecursiveVecGPU(int* data, int* result, cl_uint n, cl_context context, cl_command_queue queue, cl_kernel scanBlocks, cl_kernel addSums, size_t workGroupSize)
{
	size_t bufferSize = roundToMultiple(n, workGroupSize * 2 * VECTOR_WIDTH);

	cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize * sizeof(int), nullptr, nullptr);

	clEnqueueWriteBuffer(queue, buffer, false, 0, n * sizeof(int), data, 0, nullptr, nullptr);

	scanRecursiveVecGPU_r(buffer, bufferSize, context, queue, scanBlocks, addSums, workGroupSize);

	clEnqueueReadBuffer(queue, buffer, true, 0, n * sizeof(int), result, 0, nullptr, nullptr);

	clReleaseMemObject(buffer);
} // scanRecursiveVecGPU

string readFile(string fileName)
{
	ifstream file(fileName, ios::in);
	if(!file)
		throw runtime_error("Error opening file " + fileName);

	string buffer = string((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());

	file.close();

	return buffer;
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
	string source1 = readFile("../../../src/scan/gpu/thesis/NaiveScan.cl");
	string source2 = readFile("../../../src/scan/gpu/thesis/WorkEfficientScan.cl");
	string source3 = readFile("../../../src/scan/gpu/thesis/RecursiveScan.cl");
	string source4 = readFile("../../../src/scan/gpu/thesis/RecursiveVecScan.cl");

	const char* source1Ptr = source1.c_str();
	const char* source2Ptr = source2.c_str();
	const char* source3Ptr = source3.c_str();
	const char* source4Ptr = source4.c_str();
//
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
	cl_kernel kernel1 = clCreateKernel(program1, "ScanNaive", nullptr);
	cl_kernel kernel2_up = clCreateKernel(program2, "UpSweep", nullptr);
	cl_kernel kernel2_down = clCreateKernel(program2, "DownSweep", nullptr);
	cl_kernel kernel3_scan = clCreateKernel(program3, "ScanBlocks", nullptr);
	cl_kernel kernel3_sums = clCreateKernel(program3, "AddSums", nullptr);
	cl_kernel kernel4_scan = clCreateKernel(program4, "ScanBlocksVec", nullptr);
	cl_kernel kernel4_sums = clCreateKernel(program4, "AddSumsVec", nullptr);

	// SCAN
	size_t n = 1<<20;
	int* input = new int[n];
	int* result0 = new int[n]();
	int* result1 = new int[n]();
	int* result2 = new int[n]();
	int* result3 = new int[n]();
	int* result4 = new int[n]();

	generate(input, input + n, []()
	{
		return rand() % 100;
	});

	scanCPU(input, result0, n);

	scanNaiveGPU(input, result1, n, context, queue, kernel1, 256);
	if(memcmp(result0 + 1, result1, (n - 1) * sizeof(int))) // inclusive scan
		cerr << "validation of kernel 1 failed" << endl;
	else
		cout << "kernel 1 ok" << endl;

	scanWorkEfficientGPU(input, result2, n, context, queue, kernel2_up, kernel2_down, 256);
	if(memcmp(result0, result2, n * sizeof(int)))
		cerr << "validation of kernel 2 failed" << endl;
	else
		cout << "kernel 2 ok" << endl;

	scanRecursiveGPU(input, result3, n, context, queue, kernel3_scan, kernel3_sums, 256);
	if(memcmp(result0, result3, n * sizeof(int)))
		cerr << "validation of kernel 3 failed" << endl;
	else
		cout << "kernel 3 ok" << endl;

	scanRecursiveVecGPU(input, result4, n, context, queue, kernel4_scan, kernel4_sums, 256);
	if(memcmp(result0, result4, n * sizeof(int)))
		cerr << "validation of kernel 4 failed" << endl;
	else
		cout << "kernel 4 ok" << endl;

	delete[] input;
	delete[] result0;
	delete[] result1;
	delete[] result2;
	delete[] result3;
	delete[] result4;

	clReleaseKernel(kernel1);
	clReleaseKernel(kernel2_up);
	clReleaseKernel(kernel2_down);
	clReleaseKernel(kernel3_scan);
	clReleaseKernel(kernel3_sums);
	clReleaseKernel(kernel4_scan);
	clReleaseKernel(kernel4_sums);
	clReleaseProgram(program1);
	clReleaseProgram(program2);
	clReleaseProgram(program3);
	clReleaseProgram(program4);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	clReleaseDevice(device);

	return 0;
}
