#ifndef TILE_SIZE
#error "TILE_SIZE must be defined"
#endif

#ifndef T
#error "T must be defined"
#endif

__kernel void MultLocal(__global const T* a, __global const T* b, __global T* c, uint size)
{
    uint globalX = get_global_id(0);
    uint globalY = get_global_id(1);
    uint groupX = get_group_id(0);
    uint groupY = get_group_id(1);
    uint localX = get_local_id(0);
    uint localY = get_local_id(1);

    uint aBegin = size * (TILE_SIZE * groupY);
    uint aEnd = aBegin + size - 1;
    uint aStep = TILE_SIZE;

    uint bBegin = TILE_SIZE * groupX;
    uint bStep = TILE_SIZE * size;

    T sum = 0.0;

    for(int aPos = aBegin, bPos = bBegin; aPos <= aEnd; aPos += aStep, bPos += bStep)
    {
        __local T aTile[TILE_SIZE][TILE_SIZE];
        __local T bTile[TILE_SIZE][TILE_SIZE];

        aTile[localY][localX] = a[size * localY + localX + aPos];
        bTile[localY][localX] = b[size * localY + localX + bPos];

        barrier(CLK_LOCAL_MEM_FENCE);

        for(uint k = 0; k < TILE_SIZE; k++)
            sum += aTile[localY][k] * bTile[k][localX];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[globalY * size + globalX] = sum;
}