__kernel void MultBlock(__global T* a, __global T* b, __global T* c, uint size, __local float* aTile, __local float* bTile)
{
    int BLOCK_SIZE = get_local_size(0);

    int groupX = get_group_id(0);
    int groupY = get_group_id(1);
    int localX = get_local_id(0);
    int localY = get_local_id(1);

    int aBegin = size * (BLOCK_SIZE * groupY);
    int aEnd = aBegin + size - 1;
    int aStep = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * groupX;
    int bStep = BLOCK_SIZE * size;

    T sum = 0.0;

    for(int aPos = aBegin, bPos = bBegin; aPos <= aEnd; aPos += aStep, bPos += bStep)
    {
        aTile[localY * BLOCK_SIZE + localX] = a[size * localY + localX + aPos];
        bTile[localY * BLOCK_SIZE + localX] = b[size * localY + localX + bPos];

        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k = 0; k < BLOCK_SIZE; k++)
            sum += aTile[localY * BLOCK_SIZE + k] * bTile[k * BLOCK_SIZE + localX];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[get_global_id(1) * size + get_global_id(0)] = sum;
}