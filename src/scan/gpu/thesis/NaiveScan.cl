__kernel void NaiveScan(__global int* src, __global int* dst, uint offset, uint n)
{
    size_t id = get_global_id(0);

    if(id >= n)
        return;

    if(id >= offset)
        dst[id] = src[id] + src[id - offset];
    else if(id >= (offset >> 1))
        dst[id] = src[id];
}
