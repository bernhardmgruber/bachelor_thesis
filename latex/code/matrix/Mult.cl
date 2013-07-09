__kernel void Mult(__global const float* a, __global const float* b, __global float* c, uint size)
{
    size_t col = get_global_id(0);
    size_t row = get_global_id(1);

    if(row >= size || col >= size)
        return;

    float sum = 0;
    for(size_t i = 0; i < size; i++)
        sum += a[row * size + i] * b[i * size + col];

    c[row * size + col] = sum;
}
