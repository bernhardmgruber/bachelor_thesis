__kernel void UpSweep(__global int* buffer, uint offset)
{
    uint stride = offset << 1;

    uint id = (get_global_id(0) + 1) * stride - 1;

    buffer[id] += buffer[id - offset];
}

__kernel void DownSweep(__global int* buffer, uint offset)
{
    uint stride = offset << 1;

    uint id = (get_global_id(0) + 1) * stride - 1;

    int val = buffer[id];
    buffer[id] += buffer[id - offset];
    buffer[id - offset] = val;
}