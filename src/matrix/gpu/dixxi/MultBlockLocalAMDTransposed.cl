#ifndef BLOCK_SIZE
#error "BLOCK_SIZE must be defined"
#endif

#ifndef T4
#error "T4 must be defined"
#endif

// Matrixes A and B are cached into local memory block 
// Required global threads = (size / 4, size / 4) 
__kernel void MultBlockLocalTransposed(__global const T4 * a, __global const T4 * b, __global T4 * c, uint size, __local T4 * aTile, __local T4 * bTile)
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
        int globalPosB = ((tileIndex * get_local_size(1) + get_local_id(1)) * BLOCK_SIZE) * get_global_size(0) + get_global_id(0);


        // Load values in aTile from matrixA 
        aTile[tilePos + 0 * get_local_size(0)] = a[globalPosA + 0 * size4];
        aTile[tilePos + 1 * get_local_size(0)] = a[globalPosA + 1 * size4];
        aTile[tilePos + 2 * get_local_size(0)] = a[globalPosA + 2 * size4];
        aTile[tilePos + 3 * get_local_size(0)] = a[globalPosA + 3 * size4];

        // Load block for bTile from matrixB
        T4 bFromGlobal0 = b[globalPosB + 0 * size4]; 
        T4 bFromGlobal1 = b[globalPosB + 1 * size4];
        T4 bFromGlobal2 = b[globalPosB + 2 * size4];
        T4 bFromGlobal3 = b[globalPosB + 3 * size4];

        // Transpose block
        bTile[tilePos + 0 * get_local_size(0)] = (T4)(bFromGlobal0.s0, bFromGlobal1.s0, bFromGlobal2.s0, bFromGlobal3.s0);
        bTile[tilePos + 1 * get_local_size(0)] = (T4)(bFromGlobal0.s1, bFromGlobal1.s1, bFromGlobal2.s1, bFromGlobal3.s1);
        bTile[tilePos + 2 * get_local_size(0)] = (T4)(bFromGlobal0.s2, bFromGlobal1.s2, bFromGlobal2.s2, bFromGlobal3.s2);
        bTile[tilePos + 3 * get_local_size(0)] = (T4)(bFromGlobal0.s3, bFromGlobal1.s3, bFromGlobal2.s3, bFromGlobal3.s3);

        barrier(CLK_LOCAL_MEM_FENCE);

        // This loop runs for number of threads in horizontal direction in the tiley
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

            sum0.s0 += dot(tempA0, tempB0);
            sum0.s1 += dot(tempA0, tempB1);
            sum0.s2 += dot(tempA0, tempB2);
            sum0.s3 += dot(tempA0, tempB3);

            sum1.s0 += dot(tempA1, tempB0);
            sum1.s1 += dot(tempA1, tempB1);
            sum1.s2 += dot(tempA1, tempB2);
            sum1.s3 += dot(tempA1, tempB3);

            sum2.s0 += dot(tempA2, tempB0);
            sum2.s1 += dot(tempA2, tempB1);
            sum2.s2 += dot(tempA2, tempB2);
            sum2.s3 += dot(tempA2, tempB3);

            sum3.s0 += dot(tempA3, tempB0);
            sum3.s1 += dot(tempA3, tempB1);
            sum3.s2 += dot(tempA3, tempB2);
            sum3.s3 += dot(tempA3, tempB3);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write 16 values to matrixC
    c[globalPosC + 0 * get_global_size(0)] = sum0;
    c[globalPosC + 1 * get_global_size(0)] = sum1;
    c[globalPosC + 2 * get_global_size(0)] = sum2;
    c[globalPosC + 3 * get_global_size(0)] = sum3;
}
