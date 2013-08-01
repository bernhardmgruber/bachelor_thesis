#define CONCAT(a, b) a ## b
#define CONCAT_EXPANED(a, b) CONCAT(a, b)

#define UPSWEEP_STEP(left, right) right += left

#define DOWNSWEEP_STEP_RND(left, right, tmp) \
    int tmp = left;                          \
    left = right;                            \
    right += tmp

#define DOWNSWEEP_STEP(left, right) DOWNSWEEP_STEP_RND(left, right, CONCAT_EXPANED(tmp, __COUNTER__))

/**
* From: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
* Chapter: 39.2.2 A Work-Efficient Parallel Scan
*/
__kernel void WorkEfficientVecScan(__global int8* buffer, __global int* sums, __local int* shared)
{
    uint globalId = get_global_id(0);
    uint localId = get_local_id(0);
    uint n = get_local_size(0) * 2;

    uint offset = 1;

    //
    // load input vectors
    //

    int8 val1 = buffer[2 * globalId + 0];
    int8 val2 = buffer[2 * globalId + 1];

    //
    // scan input vectores
    //

    // upsweep
    UPSWEEP_STEP(val1.s0, val1.s1);
    UPSWEEP_STEP(val2.s0, val2.s1);
    UPSWEEP_STEP(val1.s2, val1.s3);
    UPSWEEP_STEP(val2.s2, val2.s3);
    UPSWEEP_STEP(val1.s4, val1.s5);
    UPSWEEP_STEP(val2.s4, val2.s5);
    UPSWEEP_STEP(val1.s6, val1.s7);
    UPSWEEP_STEP(val2.s6, val2.s7);

    UPSWEEP_STEP(val1.s1, val1.s3);
    UPSWEEP_STEP(val2.s1, val2.s3);
    UPSWEEP_STEP(val1.s5, val1.s7);
    UPSWEEP_STEP(val2.s5, val2.s7);

    UPSWEEP_STEP(val1.s3, val1.s7);
    UPSWEEP_STEP(val2.s3, val2.s7);

    // sums
    int sum1 = val1.s7;
    int sum2 = val2.s7;

    // move sums into shared memory
    shared[2 * localId + 0] = sum1;
    shared[2 * localId + 1] = sum2;

    // set last elements to zero
    val1.s7 = 0;
    val2.s7 = 0;

    // downsweep
    DOWNSWEEP_STEP(val1.s3, val1.s7);
    DOWNSWEEP_STEP(val2.s3, val2.s7);

    DOWNSWEEP_STEP(val1.s1, val1.s3);
    DOWNSWEEP_STEP(val2.s1, val2.s3);
    DOWNSWEEP_STEP(val1.s5, val1.s7);
    DOWNSWEEP_STEP(val2.s5, val2.s7);

    DOWNSWEEP_STEP(val1.s0, val1.s1);
    DOWNSWEEP_STEP(val2.s0, val2.s1);
    DOWNSWEEP_STEP(val1.s2, val1.s3);
    DOWNSWEEP_STEP(val2.s2, val2.s3);
    DOWNSWEEP_STEP(val1.s4, val1.s5);
    DOWNSWEEP_STEP(val2.s4, val2.s5);
    DOWNSWEEP_STEP(val1.s6, val1.s7);
    DOWNSWEEP_STEP(val2.s6, val2.s7);

    //
    // scan the sums
    //

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
    if (localId == 0)
    {
        sums[get_group_id(0)] = shared[n - 1];
        shared[n - 1] = 0;    // clear the last element
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

    //
    // apply the sums
    //

    val1 += shared[2 * localId + 0];
    val2 += shared[2 * localId + 1];

    //
    // write results to device memory
    //

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

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
//#define CONFLICT_FREE_OFFSET(addr) ((addr) >> NUM_BANKS + (addr) >> (2 * LOG_NUM_BANKS))
#define CONFLICT_FREE_OFFSET(addr) ((addr) / NUM_BANKS)

__kernel void WorkEfficientVecScanOptim(__global int8* buffer, __global int* sums, __local int* shared)
{
    uint globalId = get_global_id(0) + get_group_id(0) * get_local_size(0);
    uint localId = get_local_id(0);
    uint n = get_local_size(0) * 2;

    uint offset = 1;

    uint ai = localId;
    uint bi = localId + (n / 2);
    uint bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    uint bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    //
    // load input vectors
    //

    int8 val1 = buffer[globalId];
    int8 val2 = buffer[globalId + ( n / 2 )];

    //
    // scan input vectores
    //

    // upsweep
    UPSWEEP_STEP(val1.s0, val1.s1);
    UPSWEEP_STEP(val2.s0, val2.s1);
    UPSWEEP_STEP(val1.s2, val1.s3);
    UPSWEEP_STEP(val2.s2, val2.s3);
    UPSWEEP_STEP(val1.s4, val1.s5);
    UPSWEEP_STEP(val2.s4, val2.s5);
    UPSWEEP_STEP(val1.s6, val1.s7);
    UPSWEEP_STEP(val2.s6, val2.s7);

    UPSWEEP_STEP(val1.s1, val1.s3);
    UPSWEEP_STEP(val2.s1, val2.s3);
    UPSWEEP_STEP(val1.s5, val1.s7);
    UPSWEEP_STEP(val2.s5, val2.s7);

    UPSWEEP_STEP(val1.s3, val1.s7);
    UPSWEEP_STEP(val2.s3, val2.s7);

    // sums
    int sum1 = val1.s7;
    int sum2 = val2.s7;

    // move sums into shared memory
    shared[ai + bankOffsetA]  = sum1;
    shared[bi + bankOffsetB]  = sum2;

    // set last elements to zero
    val1.s7 =0;
    val2.s7 =0;

    // downsweep
    DOWNSWEEP_STEP(val1.s3, val1.s7);
    DOWNSWEEP_STEP(val2.s3, val2.s7);

    DOWNSWEEP_STEP(val1.s1, val1.s3);
    DOWNSWEEP_STEP(val2.s1, val2.s3);
    DOWNSWEEP_STEP(val1.s5, val1.s7);
    DOWNSWEEP_STEP(val2.s5, val2.s7);

    DOWNSWEEP_STEP(val1.s0, val1.s1);
    DOWNSWEEP_STEP(val2.s0, val2.s1);
    DOWNSWEEP_STEP(val1.s2, val1.s3);
    DOWNSWEEP_STEP(val2.s2, val2.s3);
    DOWNSWEEP_STEP(val1.s4, val1.s5);
    DOWNSWEEP_STEP(val2.s4, val2.s5);
    DOWNSWEEP_STEP(val1.s6, val1.s7);
    DOWNSWEEP_STEP(val2.s6, val2.s7);

    //
    // scan the sums
    //

    // build sum in place up the tree
    for (uint d = n >> 1; d > 0; d >>= 1)                    
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < d)
        {
            uint ai = offset * (2 * localId + 1) - 1;
            uint bi = offset * (2 * localId + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

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
    for (uint d = 1; d < n; d *= 2) 
    {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < d)
        {
            uint ai = offset * (2 * localId + 1) - 1;
            uint bi = offset * (2 * localId + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            int t = shared[ai];
            shared[ai] = shared[bi];
            shared[bi] += t;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //
    // apply the sums
    //

    val1 += shared[ai + bankOffsetA];
    val2 += shared[bi + bankOffsetB];

    //
    // write results to device memory
    //

    buffer[globalId]           = val1;
    buffer[globalId + (n / 2)] = val2;
}