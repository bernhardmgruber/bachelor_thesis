#ifndef BLOCK_SIZE
#error "BLOCK_SIZE must be defined"
#endif

#ifndef T
#error "T must be defined"
#endif

__kernel void MultLocal(__global T* A, __global T* B, __global T* C, uint size)
{
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

    for(int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
    {
        __local T aTile[BLOCK_SIZE][BLOCK_SIZE];
        __local T bTile[BLOCK_SIZE][BLOCK_SIZE];

        aTile[localY][localX] = A[size * localY + localX + a];
        bTile[localY][localX] = B[size * localY + localX + b];

        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k = 0; k < BLOCK_SIZE; k++)
            sum += aTile[localY][k] * bTile[k][localX];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[size * BLOCK_SIZE * groupY + BLOCK_SIZE * groupX + size * localY + localX] = sum;
}