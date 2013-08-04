#ifndef T
#error "T must be defined"
#endif

/**
* From: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
* Chapter: 39.2.2 A Work-Efficient Parallel Scan
*/

__kernel void UpSweep(__global T* buffer, uint offset)
{
    uint stride = offset << 1;

    uint id = (get_global_id(0) + 1) * stride - 1;

    buffer[id] += buffer[id - offset];
}

__kernel void DownSweep(__global T* buffer, uint offset)
{
    uint stride = offset << 1;

    uint id = (get_global_id(0) + 1) * stride - 1;

    T val = buffer[id];
    buffer[id] += buffer[id - offset];
    buffer[id - offset] = val;
}
