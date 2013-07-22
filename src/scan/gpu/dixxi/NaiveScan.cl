#ifndef T
#error "T must be defined"
#endif

/**
 * From: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
 * Chapter: 39.2.1 A Naive Parallel Scan
 */
__kernel void NaiveScan(__global const T* src, __global T* dest, uint power, uint size)
{
    size_t k = get_global_id(0);

    if(k >= size)
        return;

    if(k >= power)
        dest[k] = src[k] + src[k - power];
    else
        dest[k] = src[k];
}
