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

int roundToMultiple(int x, int multiple)
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

typedef uint32_t uint;

void sortC(uint* data, size_t n) {
	qsort(data, n, sizeof(uint), [] (const void* a, const void* b) -> int {
		if (*((uint*)a) < *((uint*)b))
			return -1;
		else if (*((uint*)a) > *((uint*)b))
			return 1;
		return 0;
	});
}

void sortCPP(uint* data, size_t n) {
	std::sort(data, data + n);
}

#define RADIX 16
#define BUCKETS (1 << RADIX)
#define RADIX_MASK (BUCKETS - 1)

void radixSort(uint* data, size_t n) {
	size_t histogram[BUCKETS];
	uint* aux = new uint[n];

	uint* src = data;
	uint* dst = aux;
	for(size_t bits = 0; bits < sizeof(uint) * 8 ; bits += RADIX) {
		memset(histogram, 0, BUCKETS * sizeof(size_t));

		// calculate histogram
		for(size_t i = 0; i < n; ++i) {
			uint element = src[i];
			uint pos = (element >> bits) & RADIX_MASK;
			histogram[pos]++;
		}

		// scan histogram (exclusive)
		size_t sum = 0;
		for(size_t i = 0; i < BUCKETS; ++i) {
			size_t val = histogram[i];
			histogram[i] = sum;
			sum += val;
		}

		// permute
		for(size_t i = 0; i < n; ++i) {
			uint element = src[i];
			uint pos = (element >> bits) & RADIX_MASK;
			size_t index = histogram[pos]++;
			dst[index] = src[i];
		}

		std::swap(src, dst);
	}

	if(dst != data)
		memcpy(data, dst, n * sizeof(uint));

	delete[] aux;
}

void bitonicSort(uint* data, cl_uint n, cl_context context, cl_command_queue queue,
		cl_kernel kernel, size_t workGroupSize) {
	size_t adjustedSize = roundToPowerOfTwo(n);
	cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, adjustedSize * sizeof(uint),
			nullptr, nullptr);

	clEnqueueWriteBuffer(queue, buffer, false, 0, n * sizeof(uint), data, 0, nullptr, nullptr);
	if(adjustedSize != n) {
		cl_uint max = numeric_limits<uint>::max();
		clEnqueueFillBuffer(queue, buffer, &max, sizeof(uint), n * sizeof(uint),
				(adjustedSize - n) * sizeof(uint), 0, nullptr, nullptr);
	}

	for (cl_uint boxwidth = 2; boxwidth <= adjustedSize; boxwidth <<= 1) {
		for (cl_uint inc = boxwidth >> 1; inc > 0; inc >>= 1) {
			clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
			clSetKernelArg(kernel, 1, sizeof(cl_uint), &inc);
			clSetKernelArg(kernel, 2, sizeof(cl_uint), &boxwidth);
			size_t threads = adjustedSize / 2;
			size_t globalWS[] = { threads };
			size_t localWS[] = { min(workGroupSize, threads) };
			clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, globalWS, localWS, 0, nullptr, nullptr);
		}
	}

	clEnqueueReadBuffer(queue, buffer, true, 0, n * sizeof(uint), data, 0, nullptr, nullptr);

	clReleaseMemObject(buffer);
}

void bitonicSortFusion(uint* data, cl_uint n, cl_context context, cl_command_queue queue,
		cl_kernel kernel2, cl_kernel kernel4, cl_kernel kernel8, cl_kernel kernel16, size_t workGroupSize) {
	size_t adjustedSize = roundToPowerOfTwo(n);
	cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, adjustedSize * sizeof(uint),
			nullptr, nullptr);

	clEnqueueWriteBuffer(queue, buffer, false, 0, n * sizeof(uint), data, 0, nullptr, nullptr);
	if(adjustedSize != n) {
		cl_uint max = numeric_limits<uint>::max();
		clEnqueueFillBuffer(queue, buffer, &max, sizeof(uint), n * sizeof(uint),
				(adjustedSize - n) * sizeof(uint), 0, nullptr, nullptr);
	}

	for (cl_uint boxwidth = 2; boxwidth <= adjustedSize; boxwidth <<= 1) {
		for (cl_uint inc = boxwidth >> 1; inc > 0; ) {
			int ninc = 0;
			cl_kernel kernel;

			if (inc >= 8) {
				kernel = kernel16;
				ninc = 4;
			} else if (inc >= 4) {
				kernel = kernel8;
				ninc = 3;
			} else if (inc >= 2) {
				kernel = kernel4;
				ninc = 2;
			} else {
				kernel = kernel2;
				ninc = 1;
			}

			clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
			clSetKernelArg(kernel, 1, sizeof(cl_uint), &inc);
			clSetKernelArg(kernel, 2, sizeof(cl_uint), &boxwidth);
			size_t threads = adjustedSize >> ninc;
			size_t globalWS[] = { threads };
			size_t localWS[] = { min(workGroupSize, threads) };
			clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, globalWS, localWS, 0, nullptr, nullptr);

			inc >>= ninc;
		}
	}

	clEnqueueReadBuffer(queue, buffer, true, 0, n * sizeof(uint), data, 0, nullptr, nullptr);

	clReleaseMemObject(buffer);
}

#define VECTOR_WIDTH 8

void scanCLRecursiveVector_r(cl_mem values, cl_uint n, cl_context context, cl_command_queue queue, cl_kernel scanBlocksKernel, cl_kernel addSumsKernel, size_t workGroupSize) {
	size_t sumCount = roundToMultiple(n / (workGroupSize * 2 * VECTOR_WIDTH), workGroupSize * 2 * VECTOR_WIDTH);
	cl_mem sums = clCreateBuffer(context, CL_MEM_READ_WRITE, sumCount * sizeof(uint), nullptr, nullptr);

	clSetKernelArg(scanBlocksKernel, 0, sizeof(cl_mem), &values);
	clSetKernelArg(scanBlocksKernel, 1, sizeof(cl_mem), &sums);
	clSetKernelArg(scanBlocksKernel, 2, sizeof(uint) * 2 * workGroupSize, nullptr);
	size_t globalWS[] = { n / (2 * VECTOR_WIDTH) };
	size_t localWS[] = { workGroupSize };
	clEnqueueNDRangeKernel(queue, scanBlocksKernel, 1, nullptr, globalWS, localWS, 0, nullptr, nullptr);

	if(n > workGroupSize * 2 * VECTOR_WIDTH) {
		scanCLRecursiveVector_r(sums, sumCount, context, queue, scanBlocksKernel, addSumsKernel, workGroupSize);

		clSetKernelArg(addSumsKernel, 0, sizeof(cl_mem), &values);
		clSetKernelArg(addSumsKernel, 1, sizeof(cl_mem), &sums);
		clEnqueueNDRangeKernel(queue, addSumsKernel, 1, nullptr, globalWS, localWS, 0, nullptr, nullptr);
	}

	clReleaseMemObject(sums);
}

#undef RADIX
#undef BUCKETS
#undef RADIX_MASK
#undef VECTOR_WIDTH

#define RADIX 4
#define BUCKETS (1 << RADIX)
#define RADIX_MASK (BUCKETS - 1)
#define BLOCK_SIZE 32
#define VECTOR_WIDTH 8

void radixSortCL(uint* data, cl_uint n, cl_context context, cl_command_queue queue,
		cl_kernel histogramKernel, cl_kernel permuteKernel, cl_kernel scanBlocksKernel,
		cl_kernel addSumsKernel, size_t workGroupSize) {
	size_t adjustedSize = roundToMultiple(n, workGroupSize * BLOCK_SIZE);
	cl_mem srcBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, adjustedSize * sizeof(uint),
			nullptr, nullptr);
	cl_mem dstBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, adjustedSize * sizeof(uint),
			nullptr, nullptr);

	clEnqueueWriteBuffer(queue, srcBuffer, false, 0, n * sizeof(uint), data, 0, nullptr, nullptr);
	if(adjustedSize != n) {
		cl_uint max = numeric_limits<uint>::max();
		clEnqueueFillBuffer(queue, srcBuffer, &max, sizeof(uint), n * sizeof(uint),
				(adjustedSize - n) * sizeof(uint), 0, nullptr, nullptr);
	}

	size_t histogramSize = (adjustedSize / BLOCK_SIZE) * BUCKETS;
	histogramSize = roundToMultiple(histogramSize, workGroupSize * 2 * VECTOR_WIDTH); // for scan
	cl_mem histogramBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, histogramSize * sizeof(cl_uint),
			nullptr, nullptr);

	size_t globalWS[] = { adjustedSize / BLOCK_SIZE };
	size_t localWS[] = { workGroupSize };

	for(cl_uint bits = 0; bits < sizeof(uint) * 8; bits += RADIX) {
		clSetKernelArg(histogramKernel, 0, sizeof(cl_mem), &srcBuffer);
		clSetKernelArg(histogramKernel, 1, sizeof(cl_mem), &histogramBuffer);
		clSetKernelArg(histogramKernel, 2, sizeof(cl_uint), &bits);
		clEnqueueNDRangeKernel(queue, histogramKernel, 1, nullptr, globalWS, localWS, 0, nullptr, nullptr);

		scanCLRecursiveVector_r(histogramBuffer, histogramSize, context, queue, scanBlocksKernel,
				addSumsKernel, workGroupSize);

		clSetKernelArg(permuteKernel, 0, sizeof(cl_mem), &srcBuffer);
		clSetKernelArg(permuteKernel, 1, sizeof(cl_mem), &dstBuffer);
		clSetKernelArg(permuteKernel, 2, sizeof(cl_mem), &histogramBuffer);
		clSetKernelArg(permuteKernel, 3, sizeof(cl_uint), &bits);
		clEnqueueNDRangeKernel(queue, permuteKernel, 1, nullptr, globalWS, localWS, 0, nullptr, nullptr);

		std::swap(srcBuffer, dstBuffer);
	}

	clEnqueueReadBuffer(queue, srcBuffer, true, 0, n * sizeof(uint), data, 0, nullptr, nullptr);

	clReleaseMemObject(srcBuffer);
	clReleaseMemObject(dstBuffer);
	clReleaseMemObject(histogramBuffer);
}

void radixSortCLLocal(uint* data, cl_uint n, cl_context context, cl_command_queue queue, cl_kernel histogramKernel, cl_kernel permuteKernel, cl_kernel scanBlocksKernel, cl_kernel addSumsKernel, size_t workGroupSize) {
	size_t adjustedSize = roundToMultiple(n, workGroupSize * BLOCK_SIZE);
	cl_mem srcBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, adjustedSize * sizeof(uint), nullptr, nullptr);
	cl_mem dstBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, adjustedSize * sizeof(uint), nullptr, nullptr);

	clEnqueueWriteBuffer(queue, srcBuffer, false, 0, n * sizeof(uint), data, 0, nullptr, nullptr);
	if(adjustedSize != n) {
		cl_uint max = numeric_limits<uint>::max();
		clEnqueueFillBuffer(queue, srcBuffer, &max, sizeof(uint), n * sizeof(uint), (adjustedSize - n) * sizeof(uint), 0, nullptr, nullptr);
	}

	size_t histogramSize = (adjustedSize / BLOCK_SIZE) * BUCKETS;
	histogramSize = roundToMultiple(histogramSize, workGroupSize * 2 * VECTOR_WIDTH); // for scan
	cl_mem histogramBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, histogramSize * sizeof(cl_uint), nullptr, nullptr);

	size_t localSize = (workGroupSize * BUCKETS * sizeof(cl_uint));
	size_t globalWS[] = { adjustedSize / BLOCK_SIZE };
	size_t localWS[] = { workGroupSize };

	for(cl_uint bits = 0; bits < sizeof(uint) * 8; bits += RADIX) {
		clSetKernelArg(histogramKernel, 0, sizeof(cl_mem), &srcBuffer);
		clSetKernelArg(histogramKernel, 1, sizeof(cl_mem), &histogramBuffer);
		clSetKernelArg(histogramKernel, 2, sizeof(cl_uint), &bits);
		clSetKernelArg(histogramKernel, 3, localSize, nullptr);
		clEnqueueNDRangeKernel(queue, histogramKernel, 1, nullptr, globalWS, localWS, 0, nullptr, nullptr);

		scanCLRecursiveVector_r(histogramBuffer, histogramSize, context, queue, scanBlocksKernel, addSumsKernel, workGroupSize);

		clSetKernelArg(permuteKernel, 0, sizeof(cl_mem), &srcBuffer);
		clSetKernelArg(permuteKernel, 1, sizeof(cl_mem), &dstBuffer);
		clSetKernelArg(permuteKernel, 2, sizeof(cl_mem), &histogramBuffer);
		clSetKernelArg(permuteKernel, 3, sizeof(cl_uint), &bits);
		clSetKernelArg(permuteKernel, 4, localSize, nullptr);
		clEnqueueNDRangeKernel(queue, permuteKernel, 1, nullptr, globalWS, localWS, 0, nullptr, nullptr);

		std::swap(srcBuffer, dstBuffer);
	}

	clEnqueueReadBuffer(queue, srcBuffer, true, 0, n * sizeof(uint), data, 0, nullptr, nullptr);

	clReleaseMemObject(srcBuffer);
	clReleaseMemObject(dstBuffer);
	clReleaseMemObject(histogramBuffer);
}

#if 0
	size_t localSize = (workGroupSize * BUCKETS * sizeof(cl_uint));
	...
	for(cl_uint bits = ...) {
		...
		clSetKernelArg(histogramKernel, 3, localSize, nullptr);
		...
		clSetKernelArg(permuteKernel, 4, localSize, nullptr);
#endif

#if 0
__kernel void Histogram(__global uint* data, __global uint* histograms, uint bits, __local uint* hist) {
	size_t globalId = get_global_id(0);
	size_t localId = get_local_id(0);

	hist += localId * BUCKETS;
	for(int i = 0; i < BUCKETS; ++i)
		hist[i] = 0;
	...
}

__kernel void Permute(__global uint* src, __global uint* dst, __global uint* scannedHistograms, uint bits,
		__local uint* hist) {
	size_t globalId = get_global_id(0);
	size_t localId = get_local_id(0);

	hist += localId * BUCKETS;
	...
}
#endif

#if 0
#define BLOCK_SIZE 128
#define BLOCK_SIZE_16 (BLOCK_SIZE / 16)

__kernel void Histogram(__global uint16* data, __global uint* histograms, uint bits, __local uint* hist) {
	...
	for(int i = 0; i < BLOCK_SIZE_16; ++i) {
		uint16 value = data[globalId * BLOCK_SIZE_16 + i];
		uint16 pos = (value >> bits) & RADIX_MASK;
		hist[pos.s0]++;
		hist[pos.s1]++;
		...
		hist[pos.sF]++;
	}
	...
}

__kernel void Permute(__global uint16* src, __global uint* dst, __global uint* scannedHistograms,
		uint bits, __local uint* hist) {
	...
	for(int i = 0; i < BLOCK_SIZE_16; ++i) {
		uint16 value = src[globalId * BLOCK_SIZE_16 + i];
		uint16 pos = (value >> bits) & RADIX_MASK;
		uint16 index;
		index.s0 = hist[pos.s0]++;
		index.s1 = hist[pos.s1]++;
		...
		index.sF = hist[pos.sF]++;
		dst[index.s0] = value.s0;
		dst[index.s1] = value.s1;
		...
		dst[index.sF] = value.sF;
	}
}
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
	string source1 = readFile("../../../src/sort/gpu/thesis/BitonicSort.cl");
	string source2 = readFile("../../../src/sort/gpu/thesis/BitonicSortFusion.cl");
	string source3 = readFile("../../../src/sort/gpu/thesis/RadixSort.cl");
	string source4 = readFile("../../../src/sort/gpu/thesis/RadixSortLocal.cl");

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
	cl_kernel kernel1 = clCreateKernel(program1, "BitonicSort", nullptr);
	cl_kernel kernel2_2 = clCreateKernel(program2, "BitonicSortFusion2", nullptr);
	cl_kernel kernel2_4 = clCreateKernel(program2, "BitonicSortFusion4", nullptr);
	cl_kernel kernel2_8 = clCreateKernel(program2, "BitonicSortFusion8", nullptr);
	cl_kernel kernel2_16 = clCreateKernel(program2, "BitonicSortFusion16", nullptr);
	cl_kernel kernel3_hist = clCreateKernel(program3, "Histogram", nullptr);
	cl_kernel kernel3_perm = clCreateKernel(program3, "Permute", nullptr);
	cl_kernel kernel3_scan = clCreateKernel(program3, "ScanBlocksVec", nullptr);
	cl_kernel kernel3_sums = clCreateKernel(program3, "AddSums", nullptr);
	cl_kernel kernel4_hist = clCreateKernel(program4, "Histogram", nullptr);
	cl_kernel kernel4_perm = clCreateKernel(program4, "Permute", nullptr);
	cl_kernel kernel4_scan = clCreateKernel(program4, "ScanBlocksVec", nullptr);
	cl_kernel kernel4_sums = clCreateKernel(program4, "AddSums", nullptr);

	// SCAN
	size_t n = 1<<20;
	uint* input = new uint[n];
	uint* buffer0 = new uint[n]();
	uint* buffer1 = new uint[n]();
	uint* buffer2 = new uint[n]();
	uint* buffer3 = new uint[n]();
	uint* buffer4 = new uint[n]();
	uint* buffer5 = new uint[n]();
	uint* buffer6 = new uint[n]();

	generate(input, input + n, []()
	{
		return rand() % 100;
	});

	copy(input, input + n, buffer0);
	copy(input, input + n, buffer1);
	copy(input, input + n, buffer2);
	copy(input, input + n, buffer3);
	copy(input, input + n, buffer4);
	copy(input, input + n, buffer5);
	copy(input, input + n, buffer6);

	sortCPP(buffer0, n);

	sortC(buffer1, n);
	if(memcmp(buffer0, buffer1, n * sizeof(uint)))
		cerr << "validation of qsort() failed" << endl;
	else
		cout << "qsort() ok" << endl;

	radixSort(buffer2, n);
	if(memcmp(buffer0, buffer2, n * sizeof(uint)))
		cerr << "validation of radixSort() failed" << endl;
	else
		cout << "radixSort() ok" << endl;

	bitonicSort(buffer3, n, context, queue, kernel1, 256);
	if(memcmp(buffer0, buffer3, n * sizeof(uint)))
		cerr << "validation of bitonic sort failed" << endl;
	else
		cout << "bitonic sort ok" << endl;

	bitonicSortFusion(buffer4, n, context, queue, kernel2_2, kernel2_4, kernel2_8, kernel2_16, 256);
	if(memcmp(buffer0, buffer4, n * sizeof(uint)))
		cerr << "validation of bitonic sort fusion failed" << endl;
	else
		cout << "bitonic sort fusion ok" << endl;

	radixSortCL(buffer5, n, context, queue, kernel3_hist, kernel3_perm, kernel3_scan, kernel3_sums, 256);
	if(memcmp(buffer0, buffer5, n * sizeof(uint)))
		cerr << "validation of radix sort failed" << endl;
	else
		cout << "radix sort ok" << endl;

	radixSortCLLocal(buffer6, n, context, queue, kernel4_hist, kernel4_perm, kernel4_scan, kernel4_sums, 256);
	if(memcmp(buffer0, buffer6, n * sizeof(uint)))
		cerr << "validation of radix sort local failed" << endl;
	else
		cout << "radix sort local ok" << endl;

	delete[] input;
	delete[] buffer0;
	delete[] buffer1;
	delete[] buffer2;
	delete[] buffer3;
	delete[] buffer4;
	delete[] buffer5;
	delete[] buffer6;

	clReleaseKernel(kernel1);
	clReleaseKernel(kernel2_2);
	clReleaseKernel(kernel2_4);
	clReleaseKernel(kernel2_8);
	clReleaseKernel(kernel2_16);
	clReleaseKernel(kernel3_hist);
	clReleaseKernel(kernel3_perm);
	clReleaseKernel(kernel3_scan);
	clReleaseKernel(kernel3_sums);
	clReleaseKernel(kernel4_hist);
	clReleaseKernel(kernel4_perm);
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
