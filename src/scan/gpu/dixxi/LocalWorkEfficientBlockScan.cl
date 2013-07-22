#ifndef T
#error "T must be defined"
#endif

#ifndef BLOCK_SIZE
#error "BLOCK_SIZE must be defined"
#endif

#define CONCAT(a, b) a ## b
#define CONCAT_EXPANED(a, b) CONCAT(a, b)

#define TB CONCAT_EXPANED(T, BLOCK_SIZE)

/**
 * From: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
 * Chapter: 39.2.2 A Work-Efficient Parallel Scan
 */


__kernel void WorkEfficientBlockScan(__global const TB* src, __global TB* dest, __local TB* shared)
{
    size_t globalId = get_global_id(0);
    size_t k = get_local_id(0);
    size_t n = get_local_size(0) * 2;

    int offset = 1;

    // load input into shared memory
    shared[2 * k]     = src[2 * globalId];
    shared[2 * k + 1] = src[2 * globalId + 1];

    for (int d = n >> 1; d > 0; d >>= 1)                    // build sum in place up the tree
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (k < d)
        {
            int ai = offset*(2*k+1)-1;
            int bi = offset*(2*k+2)-1;

            shared[bi] += shared[ai];
        }
        offset <<= 1;
    }

    if (k == 0)
        shared[n - 1] = 0;    // clear the last element

    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (k < d)
        {
            int ai = offset*(2*k+1)-1;
            int bi = offset*(2*k+2)-1;

            T t = shared[ai];
            shared[ai] = shared[bi];
            shared[bi] += t;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // write results to device memory
    dest[2 * globalId]     = shared[2 * k];
    dest[2 * globalId + 1] = shared[2 * k + 1];
}
