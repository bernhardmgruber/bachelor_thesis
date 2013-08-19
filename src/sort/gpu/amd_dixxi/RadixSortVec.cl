#define CONCAT(a, b) a ## b
#define CONCAT_EXPANDED(a, b) CONCAT(a, b)

#ifndef RADIX
#error "RADIX has to be defined"
#endif

#define BUCKETS (1 << RADIX)
#define RADIX_MASK (BUCKETS - 1)

#define TV CONCAT_EXPANDED(T, FETCH_VECTOR_WIDTH)
#define uintV CONCAT_EXPANDED(uint, FETCH_VECTOR_WIDTH)

#define BLOCK_VECTOR_SIZE (BLOCK_SIZE / FETCH_VECTOR_WIDTH)


__kernel void Histogram(__global const TV* data, __global T* histograms, uint bits
#ifndef THREAD_HIST_IN_REGISTERS
                        , __local uint* hist
#endif
                        )
{
    size_t globalId = get_global_id(0);

#ifdef THREAD_HIST_IN_REGISTERS
    uint hist[BUCKETS] = {0};
#else
    size_t localId = get_local_id(0);
    hist += localId * BUCKETS;

    for(int i = 0; i < BUCKETS; ++i)
        hist[i] = 0;
#endif

    for(int i = 0; i < BLOCK_VECTOR_SIZE; ++i)
    {
        TV value = data[globalId * BLOCK_VECTOR_SIZE + i];
        uintV pos = (value >> bits) & RADIX_MASK;

        hist[pos.s0]++;
        hist[pos.s1]++;
#if FETCH_VECTOR_WIDTH > 2
        hist[pos.s2]++;
        hist[pos.s3]++;
#endif
#if FETCH_VECTOR_WIDTH > 4
        hist[pos.s4]++;
        hist[pos.s5]++;
        hist[pos.s6]++;
        hist[pos.s7]++;
#endif
#if FETCH_VECTOR_WIDTH > 8
        hist[pos.s8]++;
        hist[pos.s9]++;
        hist[pos.sA]++;
        hist[pos.sB]++;
        hist[pos.sC]++;
        hist[pos.sD]++;
        hist[pos.sE]++;
        hist[pos.sF]++;
#endif
    }

    for(int i = 0; i < BUCKETS; ++i)
        histograms[get_global_size(0) * i + globalId] = hist[i];
}

__kernel void Permute(__global const TV* src, __global T* dst, __global const uint* scannedHistograms, uint bits
#ifndef THREAD_HIST_IN_REGISTERS
                      , __local uint* hist
#endif
                      )
{
    size_t globalId = get_global_id(0);

#ifdef THREAD_HIST_IN_REGISTERS
    uint hist[BUCKETS];
#else
    size_t localId = get_local_id(0);
    hist += localId * BUCKETS;
#endif

    for(int i = 0; i < BUCKETS; ++i)
        hist[i] = scannedHistograms[get_global_size(0) * i + globalId];

    for(int i = 0; i < BLOCK_VECTOR_SIZE; ++i)
    {
        TV value = src[globalId * BLOCK_VECTOR_SIZE + i];
        uintV pos = (value >> bits) & RADIX_MASK;

        dst[hist[pos.s0]++] = value.s0;
        dst[hist[pos.s1]++] = value.s1;
#if FETCH_VECTOR_WIDTH > 2
        dst[hist[pos.s2]++] = value.s2;
        dst[hist[pos.s3]++] = value.s3;
#endif
#if FETCH_VECTOR_WIDTH > 4
        dst[hist[pos.s4]++] = value.s4;
        dst[hist[pos.s5]++] = value.s5;
        dst[hist[pos.s6]++] = value.s6;
        dst[hist[pos.s7]++] = value.s7;
#endif
#if FETCH_VECTOR_WIDTH > 8
        dst[hist[pos.s8]++] = value.s8;
        dst[hist[pos.s9]++] = value.s9;
        dst[hist[pos.sA]++] = value.sA;
        dst[hist[pos.sB]++] = value.sB;
        dst[hist[pos.sC]++] = value.sC;
        dst[hist[pos.sD]++] = value.sD;
        dst[hist[pos.sE]++] = value.sE;
        dst[hist[pos.sF]++] = value.sF;
#endif
    }
}
