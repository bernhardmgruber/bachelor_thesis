__kernel void UpSweep(__global int* buffer, uint offset, uint stride)
{
    uint k = get_global_id(0);

    if((k + 1) % stride == 0)
        buffer[k] += buffer[k - offset];
}

__kernel void SetLastZero(__global int* buffer, uint index)
{
    buffer[index] = 0;
}

__kernel void DownSweep(__global int* buffer, uint offset, uint stride)
{
    uint k = get_global_id(0);

    if((k + 1) % stride == 0)
    {
        int val = buffer[k];
        buffer[k] += buffer[k - offset];
        buffer[k - offset] = val;
    }
}