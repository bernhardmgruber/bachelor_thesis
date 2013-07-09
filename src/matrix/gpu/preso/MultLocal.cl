#ifndef TILE_SIZE
#error "TILE_SIZE must be defined"
#endif

#ifndef T
#error "T must be defined"
#endif

__kernel void MultLocal(__global const T* a, __global const T* b, __global T* c, uint size)
{
    int globalX = get_global_id(0);
    int globalY = get_global_id(1);
    int groupX = get_group_id(0);
    int groupY = get_group_id(1);
    int localX = get_local_id(0);
    int localY = get_local_id(1);

    int aBegin = size * (TILE_SIZE * groupY);
    int aEnd = aBegin + size - 1;
    int aStep = TILE_SIZE;

    int bBegin = TILE_SIZE * groupX;
    int bStep = TILE_SIZE * size;

    T sum = 0.0;

    for(int aPos = aBegin, bPos = bBegin; aPos <= aEnd; aPos += aStep, bPos += bStep)
    {
        __local T aTile[TILE_SIZE][TILE_SIZE];
        __local T bTile[TILE_SIZE][TILE_SIZE];

        aTile[localY][localX] = a[size * localY + localX + aPos];
        bTile[localY][localX] = b[size * localY + localX + bPos];

        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k = 0; k < TILE_SIZE; k++)
            sum += aTile[localY][k] * bTile[k][localX];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[globalY * size + globalX] = sum;
}