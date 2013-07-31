__kernel void NaiveScan(__global int* src, __global int* dest, uint power, uint size)
{
    size_t k = get_global_id(0);

    if(k >= size)
        return;

    if(k >= power)
        dest[k] = src[k] + src[k - power];
    else if(k >= (power >> 1))
        dest[k] = src[k];
}
