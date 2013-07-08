__kernel void MultBlock(__global T* a, __global T* b, __global T* c, uint size, __local float* aTile, __local float* bTile)
{
    size_t col = get_global_id(0);
    size_t row = get_global_id(1);

    size_t groupSize = get_local_size(0);

    int aBegin = size * (groupSize * get_group_id(1));
    int aStep = groupSize;

    int bBegin = groupSize * get_group_id(0);
    int bStep = groupSize * size;

    T sum = 0;
    for (int k = 0; k < get_num_groups(0); k++)
    {
        int aIndex = aBegin + k * aStep + get_local_id(1) * size + get_local_id(0);
        int bIndex = bBegin + k * bStep + get_local_id(1) * size + get_local_id(0);
        int aTileIndex = get_local_id(1) * groupSize + get_local_id(0);
        int bTileIndex = get_local_id(1) * groupSize + get_local_id(0);

        aTile[aTileIndex] = a[aIndex];
        bTile[bTileIndex] = b[bIndex];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < groupSize; i++)
            sum += aTile[get_local_id(1) * groupSize + i] * bTile[i * groupSize + get_local_id(0)];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[row * size + col] = sum;
}