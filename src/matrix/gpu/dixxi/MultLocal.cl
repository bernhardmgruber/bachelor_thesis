__kernel void MultLocal(__global const T* a, __global const T* b, __global T* c, uint size, __local float* aTile, __local float* bTile)
{
    int tileSize = get_local_size(0);

    int groupX = get_group_id(0);
    int groupY = get_group_id(1);
    int localX = get_local_id(0);
    int localY = get_local_id(1);

    int aBegin = size * (tileSize * groupY);
    int aEnd = aBegin + size - 1;
    int aStep = tileSize;

    int bBegin = tileSize * groupX;
    int bStep = tileSize * size;

    T sum = 0.0;

    for(int aPos = aBegin, bPos = bBegin; aPos <= aEnd; aPos += aStep, bPos += bStep)
    {
        aTile[localY * tileSize + localX] = a[size * localY + localX + aPos];
        bTile[localY * tileSize + localX] = b[size * localY + localX + bPos];

        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k = 0; k < tileSize; k++)
            sum += aTile[localY * tileSize + k] * bTile[k * tileSize + localX];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[get_global_id(1) * size + get_global_id(0)] = sum;
}