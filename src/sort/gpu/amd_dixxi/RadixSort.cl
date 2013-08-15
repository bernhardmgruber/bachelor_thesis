#ifndef RADIX
#error "RADIX has to be defined"
#endif

#define BUCKETS (1 << RADIX)
#define RADIX_MASK (BUCKETS - 1)

/**
* @brief   Calculates block-histogram bin whose bin size is 256
* @param   unsortedData    array of unsorted elements
* @param   buckets         histogram buckets
* @param   bits            shift count
* @param   sharedArray     shared array for thread-histogram bins
*/
__kernel void histogram(__global const T* unsortedData, __global uint* histograms, uint bits
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

    /* Initialize shared array to zero */
    for(int i = 0; i < BUCKETS; ++i)
        hist[i] = 0;
#endif

    /* Calculate thread-histograms */
    for(int i = 0; i < BLOCK_SIZE; ++i)
    {
        T value = unsortedData[globalId * BLOCK_SIZE + i];
        value = (value >> bits) & RADIX_MASK;
        hist[value]++;
    }

    /* Copy calculated histogram bin to global memory */
    for(int i = 0; i < BUCKETS; ++i)
        histograms[get_global_size(0) * i + globalId] = hist[i];
}

/**
* @brief   Permutes the element to appropriate places based on
*          prescaned buckets values
* @param   unsortedData        array of unsorted elments
* @param   scanedBuckets       prescaned buckets for permuations
* @param   shiftCount          shift count
* @param   sharedBuckets       shared array for scaned buckets
* @param   sortedData          array for sorted elements
*/
__kernel void permute(__global const T* unsortedData, __global T* sortedData, __global const uint* scanedHistograms, uint shiftCount
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

    /* Copy prescaned thread histograms to corresponding thread shared block */
    for(int i = 0; i < BUCKETS; ++i)
        hist[i] = scanedHistograms[get_global_size(0) * i + globalId];

    /* Permute elements to appropriate location */
    for(int i = 0; i < BLOCK_SIZE; ++i)
    {
        T value = unsortedData[globalId * BLOCK_SIZE + i];
        T radix = (value >> shiftCount) & RADIX_MASK;

        uint index = hist[radix]++;

        sortedData[index] = value;
    }
}
