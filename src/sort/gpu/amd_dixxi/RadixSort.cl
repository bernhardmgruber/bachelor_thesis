#ifndef RADIX
#error "RADIX has to be defined"
#endif

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#define BUCKETS (1 << RADIX)
#define RADIX_MASK (BUCKETS - 1)

/**
 * @brief   Calculates block-histogram bin whose bin size is 256
 * @param   unsortedData    array of unsorted elements
 * @param   buckets         histogram buckets
 * @param   bits            shift count
 * @param   sharedArray     shared array for thread-histogram bins
  */
__kernel void histogram(__global const T* unsortedData, __global uint* buckets, uint bits, __local ushort* sharedArray)
{
    size_t localId = get_local_id(0);
    size_t globalId = get_global_id(0);
    size_t groupId = get_group_id(0);
    size_t groupSize = get_local_size(0);

    /* Initialize shared array to zero */
    for(int i = 0; i < BUCKETS; ++i)
        sharedArray[localId * BUCKETS + i] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    /* Calculate thread-histograms */
    for(int i = 0; i < BLOCK_SIZE; ++i)
    {
        T value = unsortedData[globalId * BLOCK_SIZE + i];
        value = (value >> bits) & RADIX_MASK;
        sharedArray[localId * BUCKETS + value]++;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    /* Copy calculated histogram bin to global memory */
    for(int i = 0; i < BUCKETS; ++i)
        buckets[globalId * BUCKETS + i] = sharedArray[localId * BUCKETS + i];
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
__kernel void permute(__global const T* unsortedData, __global const uint* scanedBuckets, uint shiftCount, __local ushort* sharedBuckets, __global T* sortedData)
{
    size_t groupId = get_group_id(0);
    size_t localId = get_local_id(0);
    size_t globalId = get_global_id(0);
    size_t groupSize = get_local_size(0);

    /* Copy prescaned thread histograms to corresponding thread shared block */
    for(int i = 0; i < BUCKETS; ++i)
        sharedBuckets[localId * BUCKETS + i] = scanedBuckets[globalId * BUCKETS + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    /* Premute elements to appropriate location */
    for(int i = 0; i < BLOCK_SIZE; ++i)
    {
        T value = unsortedData[globalId * BLOCK_SIZE + i];
        T radix = (value >> shiftCount) & RADIX_MASK;

        uint index = sharedBuckets[localId * BUCKETS + radix];

        sortedData[index] = value;

        sharedBuckets[localId * BUCKETS + radix] = index + 1;

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
