#define RADIX 4
#define BUCKETS (1 << RADIX)
#define RADIX_MASK (BUCKETS - 1)
#define BLOCK_SIZE 32

__kernel void Histogram(__global uint* data, __global uint* histograms, uint bits) {
	size_t globalId = get_global_id(0);

	uint hist[BUCKETS] = {0};
	for (int i = 0; i < BLOCK_SIZE; ++i) {
		uint value = data[globalId * BLOCK_SIZE + i];
		uint pos = (value >> bits) & RADIX_MASK;
		hist[pos]++;
	} // for

	for (int i = 0; i < BUCKETS; ++i)
		histograms[get_global_size(0) * i + globalId] = hist[i];
} // Histogram

__kernel void Permute(__global uint* src, __global uint* dst, __global uint* scannedHistograms,
		uint bits) {
	size_t globalId = get_global_id(0);

	uint hist[BUCKETS];
	for (int i = 0; i < BUCKETS; ++i)
		hist[i] = scannedHistograms[get_global_size(0) * i + globalId];

	for (int i = 0; i < BLOCK_SIZE; ++i) {
		uint value = src[globalId * BLOCK_SIZE + i];
		uint pos = (value >> bits) & RADIX_MASK;
		uint index = hist[pos]++;
		dst[index] = value;
	} // for
} // Permute

#define CONCAT(a, b) a ## b
#define CONCAT_EXPANDED(a, b) CONCAT(a, b)

#define UPSWEEP_STEP(left, right) right += left

#define UPSWEEP_STEPS(left, right) \
	UPSWEEP_STEP(CONCAT_EXPANDED(val1.s, left), CONCAT_EXPANDED(val1.s, right)); \
	UPSWEEP_STEP(CONCAT_EXPANDED(val2.s, left), CONCAT_EXPANDED(val2.s, right))

#define DOWNSWEEP_STEP_TMP(left, right, tmp) \
	int tmp = left;						  \
	left = right;							\
	right += tmp

#define DOWNSWEEP_STEP(left, right) DOWNSWEEP_STEP_TMP(left, right, CONCAT_EXPANDED(tmp, __COUNTER__))

#define DOWNSWEEP_STEPS(left, right) \
	DOWNSWEEP_STEP(CONCAT_EXPANDED(val1.s, left), CONCAT_EXPANDED(val1.s, right)); \
	DOWNSWEEP_STEP(CONCAT_EXPANDED(val2.s, left), CONCAT_EXPANDED(val2.s, right))

__kernel void ScanBlocksVec(__global int8* buffer, __global int* sums, __local int* shared)
{
	uint globalId = get_global_id(0);
	uint localId = get_local_id(0);
	uint n = get_local_size(0) * 2;

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

	// build sum in place up the tree
	for (uint d = n >> 1; d > 0; d >>= 1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (localId < d)
		{
			uint ai = offset*(2*localId+1)-1;
			uint bi = offset*(2*localId+2)-1;

			shared[bi] += shared[ai];
		}
		offset <<= 1;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// save sum and clear the last element
	if (localId == 0)
	{
		sums[get_group_id(0)] = shared[n - 1];
		shared[n - 1] = 0;
	}

	// traverse down tree & build scan
	for (uint d = 1; d < n; d *= 2)
	{
		offset >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);
		if (localId < d)
		{
			uint ai = offset*(2*localId+1)-1;
			uint bi = offset*(2*localId+2)-1;

			int t = shared[ai];
			shared[ai] = shared[bi];
			shared[bi] += t;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// apply the sums
	val1 += shared[2 * localId + 0];
	val2 += shared[2 * localId + 1];

	// write results to device memory
	buffer[2 * globalId + 0] = val1;
	buffer[2 * globalId + 1] = val2;
}

__kernel void AddSums(__global int8* buffer, __global int* sums)
{
	uint globalId = get_global_id(0);

	int val = sums[get_group_id(0)];

	buffer[globalId * 2 + 0] += val;
	buffer[globalId * 2 + 1] += val;
}
