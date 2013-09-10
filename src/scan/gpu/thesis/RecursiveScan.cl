__kernel void ScanBlocks(__global int* buffer, __global int* sums, __local int* shared) {
	uint globalId = get_global_id(0);
	uint localId  = get_local_id(0);
	uint n        = get_local_size(0) * 2;

	uint offset = 1;

	shared[2 * localId + 0] = buffer[2 * globalId + 0];
	shared[2 * localId + 1] = buffer[2 * globalId + 1];

	// build sum in place up the tree
	for (uint d = n >> 1; d > 0; d >>= 1) {
		barrier(CLK_LOCAL_MEM_FENCE);
		if (localId < d) {
			uint ai = offset*(2*localId+1)-1;
			uint bi = offset*(2*localId+2)-1;
			shared[bi] += shared[ai];
		} // if
		offset <<= 1;
	} // for
	barrier(CLK_LOCAL_MEM_FENCE);

	// save sum and clear the last element
	if (localId == 0) {
		sums[get_group_id(0)] = shared[n - 1];
		shared[n - 1] = 0;
	} // if

	// traverse down tree & build scan
	for (uint d = 1; d < n; d <<= 1) {
		offset >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);
		if (localId < d) {
			uint ai = offset*(2*localId+1)-1;
			uint bi = offset*(2*localId+2)-1;

			int t = shared[ai];
			shared[ai] = shared[bi];
			shared[bi] += t;
		} // if
	} // for
	barrier(CLK_LOCAL_MEM_FENCE);

	buffer[2 * globalId + 0] = shared[2 * localId + 0];
	buffer[2 * globalId + 1] = shared[2 * localId + 1];
} // ScanBlocks

__kernel void AddSums(__global int* buffer, __global int* sums) {
	uint globalId = get_global_id(0);

	int val = sums[get_group_id(0)];

	buffer[globalId * 2 + 0] += val;
	buffer[globalId * 2 + 1] += val;
} // AddSums

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(addr) (((addr) >> (NUM_BANKS + (addr))) >> (2 * LOG_NUM_BANKS))
//#define CONFLICT_FREE_OFFSET(addr) ((addr) / NUM_BANKS)

__kernel void ScanBlocksOptim(__global int* buffer, __global int* sums, __local int* shared)
{
	uint globalId = get_global_id(0) + get_group_id(0) * get_local_size(0);
	uint localId = get_local_id(0);
	uint n = get_local_size(0) * 2;

	uint offset = 1;

	uint ai = localId;
	uint bi = localId + (n / 2);
	uint bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	uint bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	shared[ai + bankOffsetA] = buffer[globalId];
	shared[bi + bankOffsetB] = buffer[globalId + (n / 2)];

	// build sum in place up the tree
	for (uint d = n >> 1; d > 0; d >>= 1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (localId < d)
		{
			uint ai = offset*(2*localId+1)-1;
			uint bi = offset*(2*localId+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int aa = shared[ai];
			int bb = shared[bi];

			shared[bi] += shared[ai];
		}
		offset <<= 1;
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	if (localId == 0)
	{
		uint index = n - 1 + CONFLICT_FREE_OFFSET(n - 1);
		sums[get_group_id(0)] = shared[index];
		// clear the last element
		shared[index] = 0;
	}

	// traverse down tree & build scan
	for (uint d = 1; d < n; d <<= 1)
	{
		offset >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);
		if (localId < d)
		{
			uint ai = offset*(2*localId+1)-1;
			uint bi = offset*(2*localId+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = shared[ai];
			shared[ai] = shared[bi];
			shared[bi] += t;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	int aaa = shared[ai + bankOffsetA];
	int bbb = shared[bi + bankOffsetB];
	buffer[globalId]		   = shared[ai + bankOffsetA];
	buffer[globalId + (n / 2)] = shared[bi + bankOffsetB];
}
