
#define RADIX 4
#define BUCKETS (1 << RADIX)
#define RADIX_MASK (BUCKETS - 1)

/**
 * From: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
 * Chapter: 39.2.2 A Work-Efficient Parallel Scan
 */

uint mapIndex(uint index)
{
    uint value = (index * BUCKETS) % (get_global_size(0) * 2);
    value += (index * BUCKETS) / (get_global_size(0) * 2);
    return value;
}

__kernel void LocalScan(__global T* buffer, __global T* sums, __local T* shared, const short first)
{
    size_t globalId = get_global_id(0);
    size_t thid = get_local_id(0);
    size_t n = get_local_size(0) * 2;

    int offset = 1;

    // load input into shared memory
    if(first)
    {
        shared[2 * thid]     = buffer[mapIndex(2 * globalId)];
        shared[2 * thid + 1] = buffer[mapIndex(2 * globalId + 1)];
    }
    else
    {
        shared[2 * thid]     = buffer[2 * globalId];
        shared[2 * thid + 1] = buffer[2 * globalId + 1];
    }

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

    // write results to device memory
    if(first)
    {
        buffer[mapIndex(2 * globalId)]     = shared[2 * thid];
        buffer[mapIndex(2 * globalId + 1)] = shared[2 * thid + 1];
    }
    else
    {
        buffer[2 * globalId]     = shared[2 * thid];
        buffer[2 * globalId + 1] = shared[2 * thid + 1];
    }
}

__kernel void AddSums(__global T* buffer, __global T* sums, __local T* shared, const short first)
{
    //size_t gid = get_group_id(0);
    size_t groupIdWithOffset = get_global_id(0) / get_local_size(0);
    size_t groupId = get_group_id(0);

    if(get_local_id(0) == 0)
        // the first thread of the work group loads the sum value
        shared[groupId] = sums[groupIdWithOffset];
    barrier(CLK_LOCAL_MEM_FENCE);

    size_t id = get_global_id(0);

    if(first)
    {
        buffer[mapIndex(id * 2)]     += shared[groupId];
        buffer[mapIndex(id * 2 + 1)] += shared[groupId];
    }
    else
    {
        buffer[id * 2]     += shared[groupId];
        buffer[id * 2 + 1] += shared[groupId];
    }
}

