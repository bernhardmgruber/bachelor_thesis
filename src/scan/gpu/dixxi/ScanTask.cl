#ifndef T
#error "T must be defined"
#endif

__kernel void ScanTask(__global const T* data, __global T* result, const uint size)
{
    result[0] = data[0];
    for(size_t i = 1; i < size; i++)
        result[i] = result[i - 1] + data[i];
}
