
/**
 * From: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
 * Chapter: 39.2.2 A Work-Efficient Parallel Scan
 */

__kernel void LocalScan(__global T* buffer, __global T* sums, __local T* shared)
{
    size_t globalId = get_global_id(0);
    size_t thid = get_local_id(0);
    size_t n = get_local_size(0) * 2;

    int offset = 1;

    shared[2*thid] = buffer[2*globalId]; // load input into shared memory
    shared[2*thid+1] = buffer[2*globalId+1];

    for (int d = n>>1; d > 0; d >>= 1)                    // build sum in place up the tree
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            shared[bi] += shared[ai];
        }
        offset <<= 1;
    }

    if (thid == 0)
    {
        sums[get_group_id(0)] = shared[n - 1];
        shared[n - 1] = 0;    // clear the last element
    }

    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            T t = shared[ai];
            shared[ai] = shared[bi];
            shared[bi] += t;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    buffer[2*globalId] = shared[2*thid]; // write results to device memory
    buffer[2*globalId+1] = shared[2*thid+1];
}

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

__kernel void LocalScanOptim(__global T* buffer, __global T* sums, __local T* shared)
{
    size_t globalId = get_global_id(0) + get_group_id(0) * get_local_size(0);
    size_t thid = get_local_id(0);
    size_t n = get_local_size(0) * 2;

    int offset = 1;

    int ai = thid;
    int bi = thid + (n/2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    shared[ai + bankOffsetA] = buffer[globalId];
    shared[bi + bankOffsetB] = buffer[globalId + (n/2)];

    for (int d = n>>1; d > 0; d >>= 1)                    // build sum in place up the tree
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            shared[bi] += shared[ai];
        }
        offset <<= 1;
    }

    if (thid == 0)
    {
        int index = n - 1 + CONFLICT_FREE_OFFSET(n - 1);
        sums[get_group_id(0)] = shared[index];
        shared[index] = 0;    // clear the last element
    }

    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            T t = shared[ai];
            shared[ai] = shared[bi];
            shared[bi] += t;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    buffer[globalId] = shared[ai + bankOffsetA];
    buffer[globalId + (n/2)] = shared[bi + bankOffsetB];
}

__kernel void AddSums(__global T* buffer, __global T* sums, __local T* shared)
{
    //size_t gid = get_group_id(0);
    size_t groupIdWithOffset = get_global_id(0) / get_local_size(0);
    size_t groupId = get_group_id(0);

    if(get_local_id(0) == 0)
        // the first thread of the work group loads the sum value
        shared[groupId] = sums[groupIdWithOffset];
    barrier(CLK_LOCAL_MEM_FENCE);

    size_t id = get_global_id(0);

    buffer[id * 2]     += shared[groupId];
    buffer[id * 2 + 1] += shared[groupId];
}

