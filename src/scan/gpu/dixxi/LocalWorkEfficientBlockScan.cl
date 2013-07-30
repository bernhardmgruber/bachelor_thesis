#ifndef T
#error "T must be defined"
#endif

#ifndef BLOCK_SIZE
#error "BLOCK_SIZE must be defined"
#endif

#ifndef BLOCK_SIZE_MINUS_ONE
#error "BLOCK_SIZE_MINUS_ONE must be defined"
#endif

#if BLOCK_SIZE < 2
#error "BLOCK_SIZE must be at least 2"
#endif

#define CONCAT(a, b) a ## b
#define CONCAT_EXPANED(a, b) CONCAT(a, b)

#define TB CONCAT_EXPANED(T, BLOCK_SIZE)

#define VECTOR_ELEMENT(v, e) CONCAT_EXPANED(v.s, e)

#define UPSWEEP_STEP(left, right) right += left

#define DOWNSWEEP_STEP_RND(left, right, tmp) \
    T tmp = left;                            \
    left = right;                            \
    right += tmp

#define DOWNSWEEP_STEP(left, right) DOWNSWEEP_STEP_RND(left, right, CONCAT_EXPANED(tmp, __COUNTER__))

/**
* From: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
* Chapter: 39.2.2 A Work-Efficient Parallel Scan
*/
__kernel void WorkEfficientBlockScan(__global TB* buffer, __local T* shared)
{
    size_t globalId = get_global_id(0);
    size_t localId = get_local_id(0);
    size_t n = get_local_size(0) * 2;

    uint offset = 1;

    //
    // load input vectors
    //

    TB val1 = buffer[2 * globalId + 0];
    TB val2 = buffer[2 * globalId + 1];

    //
    // scan input vectores
    //

    // upsweep
    UPSWEEP_STEP(val1.s0, val1.s1);
    UPSWEEP_STEP(val2.s0, val2.s1);
#if BLOCK_SIZE > 2
    UPSWEEP_STEP(val1.s2, val1.s3);
    UPSWEEP_STEP(val2.s2, val2.s3);
#endif
#if BLOCK_SIZE > 4
    UPSWEEP_STEP(val1.s4, val1.s5);
    UPSWEEP_STEP(val2.s4, val2.s5);
    UPSWEEP_STEP(val1.s6, val1.s7);
    UPSWEEP_STEP(val2.s6, val2.s7);
#endif
#if BLOCK_SIZE > 8
    UPSWEEP_STEP(val1.s8, val1.s9);
    UPSWEEP_STEP(val2.s8, val2.s9);
    UPSWEEP_STEP(val1.s10, val1.s11);
    UPSWEEP_STEP(val2.s10, val2.s11);
    UPSWEEP_STEP(val1.s12, val1.s13);
    UPSWEEP_STEP(val2.s12, val2.s13);
    UPSWEEP_STEP(val1.s14, val1.s15);
    UPSWEEP_STEP(val2.s14, val2.s15);
#endif

#if BLOCK_SIZE > 2
    UPSWEEP_STEP(val1.s1, val1.s3);
    UPSWEEP_STEP(val2.s1, val2.s3);
#endif
#if BLOCK_SIZE > 4
    UPSWEEP_STEP(val1.s5, val1.s7);
    UPSWEEP_STEP(val2.s5, val2.s7);
#endif
#if BLOCK_SIZE > 8
    UPSWEEP_STEP(val1.s9, val1.s11);
    UPSWEEP_STEP(val2.s9, val2.s11);
    UPSWEEP_STEP(val1.s13, val1.s15);
    UPSWEEP_STEP(val2.s13, val2.s15);
#endif

#if BLOCK_SIZE > 4
    UPSWEEP_STEP(val1.s3, val1.s7);
    UPSWEEP_STEP(val2.s3, val2.s7);
#endif
#if BLOCK_SIZE > 8
    UPSWEEP_STEP(val1.s11, val1.s15);
    UPSWEEP_STEP(val2.s11, val2.s15);
#endif

#if BLOCK_SIZE > 8
    UPSWEEP_STEP(val1.s7, val1.s15);
    UPSWEEP_STEP(val2.s7, val2.s15);
#endif

    // sums
    T sum1 = VECTOR_ELEMENT(val1, BLOCK_SIZE_MINUS_ONE);
    T sum2 = VECTOR_ELEMENT(val2, BLOCK_SIZE_MINUS_ONE);

    // move sums into shared memory
    shared[2 * localId + 0] = sum1;
    shared[2 * localId + 1] = sum2;

    // set last elements to zero
    VECTOR_ELEMENT(val1, BLOCK_SIZE_MINUS_ONE) = 0;
    VECTOR_ELEMENT(val2, BLOCK_SIZE_MINUS_ONE) = 0;

    // downsweep
#if BLOCK_SIZE > 8
    DOWNSWEEP_STEP(val1.s7, val1.s15);
    DOWNSWEEP_STEP(val2.s7, val2.s15);
#endif

#if BLOCK_SIZE > 4
    DOWNSWEEP_STEP(val1.s3, val1.s7);
    DOWNSWEEP_STEP(val2.s3, val2.s7);
#endif
#if BLOCK_SIZE > 8
    DOWNSWEEP_STEP(val1.s11, val1.s15);
    DOWNSWEEP_STEP(val2.s11, val2.s15);
#endif

#if BLOCK_SIZE > 2
    DOWNSWEEP_STEP(val1.s1, val1.s3);
    DOWNSWEEP_STEP(val2.s1, val2.s3);
#endif
#if BLOCK_SIZE > 4
    DOWNSWEEP_STEP(val1.s5, val1.s7);
    DOWNSWEEP_STEP(val2.s5, val2.s7);
#endif
#if BLOCK_SIZE > 8
    DOWNSWEEP_STEP(val1.s9, val1.s11);
    DOWNSWEEP_STEP(val2.s9, val2.s11);
    DOWNSWEEP_STEP(val1.s13, val1.s15);
    DOWNSWEEP_STEP(val2.s13, val2.s15);
#endif

    DOWNSWEEP_STEP(val1.s0, val1.s1);
    DOWNSWEEP_STEP(val2.s0, val2.s1);
#if BLOCK_SIZE > 2
    DOWNSWEEP_STEP(val1.s2, val1.s3);
    DOWNSWEEP_STEP(val2.s2, val2.s3);
#endif
#if BLOCK_SIZE > 4
    DOWNSWEEP_STEP(val1.s4, val1.s5);
    DOWNSWEEP_STEP(val2.s4, val2.s5);
    DOWNSWEEP_STEP(val1.s6, val1.s7);
    DOWNSWEEP_STEP(val2.s6, val2.s7);
#endif
#if BLOCK_SIZE > 8
    DOWNSWEEP_STEP(val1.s8, val1.s9);
    DOWNSWEEP_STEP(val2.s8, val2.s9);
    DOWNSWEEP_STEP(val1.s10, val1.s11);
    DOWNSWEEP_STEP(val2.s10, val2.s11);
    DOWNSWEEP_STEP(val1.s12, val1.s13);
    DOWNSWEEP_STEP(val2.s12, val2.s13);
    DOWNSWEEP_STEP(val1.s14, val1.s15);
    DOWNSWEEP_STEP(val2.s14, val2.s15);
#endif

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

    // clear the last element
    if (localId == 0)
        shared[n - 1] = 0;    

    // traverse down tree & build scan
    for (uint d = 1; d < n; d *= 2) 
    {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < d)
        {
            uint ai = offset*(2*localId+1)-1;
            uint bi = offset*(2*localId+2)-1;

            T t = shared[ai];
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
