__kernel void Mult(__global const T* a, __global const T* b, __global T* c, uint size)
{
    if(get_global_id(0) >= size * size)
        return;

    size_t col = get_global_id(0) % size;
    size_t row = get_global_id(0) / size;

    T sum = 0;
    for(size_t i = 0; i < size; i++)
        sum += a[row * size + i] * b[i * size + col];

    c[get_global_id(0)] = sum;
}
