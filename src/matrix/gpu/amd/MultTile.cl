#ifndef BLOCK_SIZE
#error "BLOCK_SIZE must be defined"
#endif

#ifndef T4
#error "T4 must be defined"
#endif

// Output tile size : 4x4 = Each thread computes 16 float values
// Required global threads = (widthC / 4, heightC / 4) 
// This kernel runs on 7xx and CPU as they don't have hardware local memory 
__kernel void MultTile(__global const T4* a, __global const T4* b, __global T4* c, uint size)
{
    int2 pos = (int2)(get_global_id(0), get_global_id(1));

    // Vectorization of input Matrices reduces their width by a factor of 4 
    uint size4 = size / BLOCK_SIZE;

    if(pos.x >= size4 || pos.y >= size4)
        return;

    T4 sum0 = (T4)(0);
    T4 sum1 = (T4)(0);
    T4 sum2 = (T4)(0);
    T4 sum3 = (T4)(0);

    for(int i = 0; i < size; i = i + BLOCK_SIZE)
    {
        T4 aBlock0 = a[i / BLOCK_SIZE + ((pos.y * BLOCK_SIZE) + 0) * size4];
        T4 aBlock1 = a[i / BLOCK_SIZE + ((pos.y * BLOCK_SIZE) + 1) * size4];
        T4 aBlock2 = a[i / BLOCK_SIZE + ((pos.y * BLOCK_SIZE) + 2) * size4];
        T4 aBlock3 = a[i / BLOCK_SIZE + ((pos.y * BLOCK_SIZE) + 3) * size4];

        // Matrix B is not transposed
        T4 bBlock0 = b[pos.x + (i + 0) * size4];
        T4 bBlock1 = b[pos.x + (i + 1) * size4];
        T4 bBlock2 = b[pos.x + (i + 2) * size4];
        T4 bBlock3 = b[pos.x + (i + 3) * size4];

        sum0.x += aBlock0.x * bBlock0.x + aBlock0.y * bBlock1.x + aBlock0.z * bBlock2.x + aBlock0.w * bBlock3.x;
        sum0.y += aBlock0.x * bBlock0.y + aBlock0.y * bBlock1.y + aBlock0.z * bBlock2.y + aBlock0.w * bBlock3.y;
        sum0.z += aBlock0.x * bBlock0.z + aBlock0.y * bBlock1.z + aBlock0.z * bBlock2.z + aBlock0.w * bBlock3.z;
        sum0.w += aBlock0.x * bBlock0.w + aBlock0.y * bBlock1.w + aBlock0.z * bBlock2.w + aBlock0.w * bBlock3.w;

        sum1.x += aBlock1.x * bBlock0.x + aBlock1.y * bBlock1.x + aBlock1.z * bBlock2.x + aBlock1.w * bBlock3.x;
        sum1.y += aBlock1.x * bBlock0.y + aBlock1.y * bBlock1.y + aBlock1.z * bBlock2.y + aBlock1.w * bBlock3.y;
        sum1.z += aBlock1.x * bBlock0.z + aBlock1.y * bBlock1.z + aBlock1.z * bBlock2.z + aBlock1.w * bBlock3.z;
        sum1.w += aBlock1.x * bBlock0.w + aBlock1.y * bBlock1.w + aBlock1.z * bBlock2.w + aBlock1.w * bBlock3.w;

        sum2.x += aBlock2.x * bBlock0.x + aBlock2.y * bBlock1.x + aBlock2.z * bBlock2.x + aBlock2.w * bBlock3.x;
        sum2.y += aBlock2.x * bBlock0.y + aBlock2.y * bBlock1.y + aBlock2.z * bBlock2.y + aBlock2.w * bBlock3.y;
        sum2.z += aBlock2.x * bBlock0.z + aBlock2.y * bBlock1.z + aBlock2.z * bBlock2.z + aBlock2.w * bBlock3.z;
        sum2.w += aBlock2.x * bBlock0.w + aBlock2.y * bBlock1.w + aBlock2.z * bBlock2.w + aBlock2.w * bBlock3.w;

        sum3.x += aBlock3.x * bBlock0.x + aBlock3.y * bBlock1.x + aBlock3.z * bBlock2.x + aBlock3.w * bBlock3.x;
        sum3.y += aBlock3.x * bBlock0.y + aBlock3.y * bBlock1.y + aBlock3.z * bBlock2.y + aBlock3.w * bBlock3.y;
        sum3.z += aBlock3.x * bBlock0.z + aBlock3.y * bBlock1.z + aBlock3.z * bBlock2.z + aBlock3.w * bBlock3.z;
        sum3.w += aBlock3.x * bBlock0.w + aBlock3.y * bBlock1.w + aBlock3.z * bBlock2.w + aBlock3.w * bBlock3.w;
    }

    c[pos.x + ((pos.y * BLOCK_SIZE) + 0) * size4] = sum0;
    c[pos.x + ((pos.y * BLOCK_SIZE) + 1) * size4] = sum1;
    c[pos.x + ((pos.y * BLOCK_SIZE) + 2) * size4] = sum2;
    c[pos.x + ((pos.y * BLOCK_SIZE) + 3) * size4] = sum3;
}
