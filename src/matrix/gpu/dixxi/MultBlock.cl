__kernel void MultBlock(__global T* a, __global T* b, __global T* c, uint size, __local float* aTile, __local float* bTile)
{
    size_t col = get_global_id(0);
    size_t row = get_global_id(1);

    // check bounds
    if(col >= size || row >= size)
        return;

    size_t groupSize = get_global_size(0) / get_num_groups(0);

    T sum = 0;
    for (int k = 0; k < size; k += groupSize)
    {
        aTile[get_local_id(1) * groupSize + get_local_id(0)] = a[row * groupSize + get_local_id(0)];
        bTile[get_local_id(1) * groupSize + get_local_id(0)] = b[get_local_id(1) * size + col];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = k; i < k + groupSize; i++)
            sum += aTile[get_local_id(1) * groupSize + i] * bTile[i * groupSize + get_local_id(0)];
    }

    c[row * size + col] = sum;
}
