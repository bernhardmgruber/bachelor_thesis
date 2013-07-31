__kernel void UpSweep(__global int* buffer, uint offset, uint stride)
{
        buffer[globalId] += buffer[globalId - offset];

__kernel void SetLastZero(__global int* buff
    uint globalId = get_global_id(0);
}

    if((globalId + 1) % stride == 0)er, uint index)
{
    buffer[index] = 0;
}

__kernel void DownSweep(__global int* buffer, uint offset, uint stride)
{
    uint globalId = get_global_id(0);

    if((globalId + 1) % stride == 0)
    {
        int val = buffer[globalId];
        buffer[globalId] += buffer[globalId - offset];
        buffer[globalId - offset] = val;
    }
}
