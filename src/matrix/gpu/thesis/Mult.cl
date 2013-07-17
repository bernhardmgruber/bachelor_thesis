__kernel void Mult(__global float* a, __global float* b, __global float* c, uint size)
{
    uint col = get_global_id(0);
    uint row = get_global_id(1);

    if(row >= size || col >= size)
        return;

    float sum = 0.0f;
    for(uint i = 0; i < size; i++)
        sum += a[row * size + i] * b[i * size + col];

    c[row * size + col] = sum;
}
