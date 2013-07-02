#ifndef BLOCK_SIZE
#error "BLOCK_SIZE must be defined"
#endif

#ifndef T4
#error "T4 must be defined"
#endif

// Matrix A is cached into local memory block 
// Required global threads = (size / 4, size / 4) 
__kernel void MultTileLocal(__global const T4 * a, __global const T4 * b, __global T4 * c, uint size, __local T4 * aTile, __local T4 * bTile)
{
    int tilePos = get_local_id(0) + (get_local_id(1) * BLOCK_SIZE) * get_local_size(0);

    // Position of thread will be according to the number of values it writes i.e TILE size 
    int globalPosC = (get_global_id(1) * BLOCK_SIZE) * get_global_size(0) + get_global_id(0);

    // each thread writes 4x4 values
    T4 sum0 = (T4)(0);
    T4 sum1 = (T4)(0);
    T4 sum2 = (T4)(0);
    T4 sum3 = (T4)(0);

    int size4 = size / 4;

    // This loop runs for the number of workgroup tiles (tiles of blocks) of A in horizontal direction
    for(int tileIndex = 0; tileIndex < (size4 / get_local_size(0)); tileIndex++)
    {
        // Calculate global ids of threads from the particular block to load from matrix A depending on tileIndex
        int globalPosA = (get_global_id(1) * BLOCK_SIZE) * get_global_size(0) + tileIndex * get_local_size(0) + get_local_id(0);

        // Calculate global ids of threads from the particular block to load from matrix B depending on tileIndex
        int globalPosB = get_global_id(0) + ((tileIndex * get_local_size(0)) * BLOCK_SIZE) * get_global_size(0);

        // Load values in aTile from matrixA 
        aTile[tilePos + 0 * get_local_size(0)] = a[globalPosA + 0 * size4];
        aTile[tilePos + 1 * get_local_size(0)] = a[globalPosA + 1 * size4];
        aTile[tilePos + 2 * get_local_size(0)] = a[globalPosA + 2 * size4];
        aTile[tilePos + 3 * get_local_size(0)] = a[globalPosA + 3 * size4];

        // Load values in bTile from matrixB
        bTile[tilePos + 0 * get_local_size(0)] = b[globalPosB + 0 * size4]; 
        bTile[tilePos + 1 * get_local_size(0)] = b[globalPosB + 1 * size4];
        bTile[tilePos + 2 * get_local_size(0)] = b[globalPosB + 2 * size4];
        bTile[tilePos + 3 * get_local_size(0)] = b[globalPosB + 3 * size4];

        barrier(CLK_LOCAL_MEM_FENCE);

        // This loop runs for number of threads in horizontal direction in the block of A 
        for(int j = 0; j < get_local_size(0); j++)
        {
            // Load 4 T4s from aTile : access patters = strided from local memory 
            T4 tempA0 = aTile[j + (get_local_id(1) * BLOCK_SIZE + 0) * get_local_size(0)];
            T4 tempA1 = aTile[j + (get_local_id(1) * BLOCK_SIZE + 1) * get_local_size(0)];
            T4 tempA2 = aTile[j + (get_local_id(1) * BLOCK_SIZE + 2) * get_local_size(0)];
            T4 tempA3 = aTile[j + (get_local_id(1) * BLOCK_SIZE + 3) * get_local_size(0)];

            // Load 4 T4s from bTile : access patters = strided from local memory 
            T4 tempB0 = bTile[get_local_id(0) + (j * BLOCK_SIZE + 0) * get_local_size(0)];
            T4 tempB1 = bTile[get_local_id(0) + (j * BLOCK_SIZE + 1) * get_local_size(0)];
            T4 tempB2 = bTile[get_local_id(0) + (j * BLOCK_SIZE + 2) * get_local_size(0)];
            T4 tempB3 = bTile[get_local_id(0) + (j * BLOCK_SIZE + 3) * get_local_size(0)];

            sum0.x += tempA0.x * tempB0.x + tempA0.y * tempB1.x + tempA0.z * tempB2.x + tempA0.w * tempB3.x;
            sum0.y += tempA0.x * tempB0.y + tempA0.y * tempB1.y + tempA0.z * tempB2.y + tempA0.w * tempB3.y;
            sum0.z += tempA0.x * tempB0.z + tempA0.y * tempB1.z + tempA0.z * tempB2.z + tempA0.w * tempB3.z;
            sum0.w += tempA0.x * tempB0.w + tempA0.y * tempB1.w + tempA0.z * tempB2.w + tempA0.w * tempB3.w;

            sum1.x += tempA1.x * tempB0.x + tempA1.y * tempB1.x + tempA1.z * tempB2.x + tempA1.w * tempB3.x;
            sum1.y += tempA1.x * tempB0.y + tempA1.y * tempB1.y + tempA1.z * tempB2.y + tempA1.w * tempB3.y;
            sum1.z += tempA1.x * tempB0.z + tempA1.y * tempB1.z + tempA1.z * tempB2.z + tempA1.w * tempB3.z;
            sum1.w += tempA1.x * tempB0.w + tempA1.y * tempB1.w + tempA1.z * tempB2.w + tempA1.w * tempB3.w;

            sum2.x += tempA2.x * tempB0.x + tempA2.y * tempB1.x + tempA2.z * tempB2.x + tempA2.w * tempB3.x;
            sum2.y += tempA2.x * tempB0.y + tempA2.y * tempB1.y + tempA2.z * tempB2.y + tempA2.w * tempB3.y;
            sum2.z += tempA2.x * tempB0.z + tempA2.y * tempB1.z + tempA2.z * tempB2.z + tempA2.w * tempB3.z;
            sum2.w += tempA2.x * tempB0.w + tempA2.y * tempB1.w + tempA2.z * tempB2.w + tempA2.w * tempB3.w;

            sum3.x += tempA3.x * tempB0.x + tempA3.y * tempB1.x + tempA3.z * tempB2.x + tempA3.w * tempB3.x;
            sum3.y += tempA3.x * tempB0.y + tempA3.y * tempB1.y + tempA3.z * tempB2.y + tempA3.w * tempB3.y;
            sum3.z += tempA3.x * tempB0.z + tempA3.y * tempB1.z + tempA3.z * tempB2.z + tempA3.w * tempB3.z;
            sum3.w += tempA3.x * tempB0.w + tempA3.y * tempB1.w + tempA3.z * tempB2.w + tempA3.w * tempB3.w;

        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write 16 values to matrixC
    c[globalPosC + 0 * get_global_size(0)] = sum0;
    c[globalPosC + 1 * get_global_size(0)] = sum1;
    c[globalPosC + 2 * get_global_size(0)] = sum2;
    c[globalPosC + 3 * get_global_size(0)] = sum3;
}
