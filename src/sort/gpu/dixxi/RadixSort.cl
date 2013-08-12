
#define RADIX_MASK (BUCKETS - 1)

#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

__kernel void Histogram(__global const T* data, __global uint* histogram, __local uint* localHistogram, const uint bits)
{
    size_t id = get_global_id(0);

    // zero local histogram
    if(id < BUCKETS)
        localHistogram[id] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // process elements and update local histogram
    T element = data[id];
    atomic_inc(&localHistogram[(element >> bits) & RADIX_MASK]);

    // push local histogram to global
    if(id < BUCKETS)
        atomic_add(&histogram[id], localHistogram[id]);
}

__kernel void Scan(__global uint* histogram)
{
    uint sum = 0;
    for(size_t i = 0; i < BUCKETS; ++i)
    {
        uint val = atomic_xchg(&histogram[i], sum);
        sum += val;
    }
}

__kernel void Permute(__global const T* data, __global T* result, __global uint* histogram, const uint bits)
{
    size_t id = get_global_id(0);

    T element = data[id];
    uint index = atomic_inc(&histogram[(element >> bits) & RADIX_MASK]);
    result[index] = element;
}
