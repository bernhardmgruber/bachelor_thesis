
__kernel void ScanTask(__global __read_only T* data, __global __write_only T* result, const uint size)
{
    result[0] = data[0];
    for(size_t i = 1; i < size; i++)
        result[i] = result[i - 1] + data[i];
}
