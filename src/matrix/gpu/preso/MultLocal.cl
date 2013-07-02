__kernel void MultLocal(__global T* A, __global T* B, __global T* C, uint size)
{
    int bx = get_group_id(0);
    int by = get_group_id(1);
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    int aBegin = size * BLOCK_SIZE * by;
    int aEnd = aBegin + size - 1;
    int aStep = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx;
    int bStep = BLOCK_SIZE * size;

    float Csub = 0.0;

    for(int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
    {
        __local T as[BLOCK_SIZE][BLOCK_SIZE];
        __local T bs[BLOCK_SIZE][BLOCK_SIZE];

        as[ty][tx] = A[a + size * ty + tx];
        bs[ty][tx] = B[b + size * ty + tx];

        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k = 0; k < BLOCK_SIZE; k++)
            Csub += as[ty][k] * bs[k][tx];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int c = size * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + size * ty + tx] = Csub;
}