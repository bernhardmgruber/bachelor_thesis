__kernel void Mult(__global const T* a, __global const T* b, __global T* c, uint size)
{
    uint col = get_global_id(0);
    uint row = get_global_id(1);

    // check bounds
    if(row >= size || col >= size)
        return;

    T sum = 0;
    for(uint i = 0; i < size; i++)
        sum += a[row * size + i] * b[i * size + col];

    c[row * size + col] = sum;
}
