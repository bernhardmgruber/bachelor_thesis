#define TILE_SIZE 16

__kernel void MultLocal(__global float* a, __global float* b, __global float* c, uint size)
{
    uint col = get_global_id(0);
    uint row = get_global_id(1);
    uint localX = get_local_id(0);
    uint localY = get_local_id(1);

    uint posA = row * size + localX;
    uint posB = localY * size + col;
    uint stepA = TILE_SIZE;
    uint stepB = TILE_SIZE * size;
    uint endA = posA + size;

    uint tilePos = localY * TILE_SIZE + localX;

    float sum = 0.0f;

    while(posA < endA)
    {
        __local float tileA[TILE_SIZE * TILE_SIZE];
        __local float tileB[TILE_SIZE * TILE_SIZE];

        tileA[tilePos] = a[posA];
        tileB[tilePos] = b[posB];

        barrier(CLK_LOCAL_MEM_FENCE);

        for(uint k = 0; k < TILE_SIZE; k++)
            sum += tileA[localY * TILE_SIZE + k] * tileB[k * TILE_SIZE + localX];

        barrier(CLK_LOCAL_MEM_FENCE);

        posA += stepA;
        posB += stepB;
    }

    c[row * size + col] = sum;
}
