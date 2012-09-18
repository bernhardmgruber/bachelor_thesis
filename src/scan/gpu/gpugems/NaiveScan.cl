
/**
 * From: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
 * Chapter: 39.2.1 A Naive Parallel Scan
 */
__kernel void NaiveScan(__global T* src, __global T* dest, uint dpower)
{
    size_t k = get_global_id(0);

    if(k >= dpower)
        dest[k] += src[k - dpower];
    else
        dest[k] = src[k];
}
