#define TILEX 4
#define TILEX_SHIFT 2
#define TILEY 4
#define TILEY_SHIFT 2

/* Output tile size : 4x4 = Each thread computes 16 float values*/
/* Required global threads = (widthC / 4, heightC / 4) */
/* This kernel runs on 7xx and CPU as they don't have hardware local memory */
__kernel void MultTile(__global T4* a, __global T4* b, __global T4* c, uint size)
{
    int2 pos = (int2)(get_global_id(0), get_global_id(1));

    if(pos.x >= size / 4 || pos.y >= size / 4)
        return;

    T4 sum0 = (T4)(0);
    T4 sum1 = (T4)(0);
    T4 sum2 = (T4)(0);
    T4 sum3 = (T4)(0);

    /* Vectorization of input Matrices reduces their width by a factor of 4 */
    uint size4 = size / 4;

    for(int i = 0; i < size4; i = i + 4)
    {
        T4 tempA0 = a[i / 4 + ((pos.y << TILEY_SHIFT) + 0) * size4];
        T4 tempA1 = a[i / 4 + ((pos.y << TILEY_SHIFT) + 1) * size4];
        T4 tempA2 = a[i / 4 + ((pos.y << TILEY_SHIFT) + 2) * size4];
        T4 tempA3 = a[i / 4 + ((pos.y << TILEY_SHIFT) + 3) * size4];

        //Matrix B is not transposed
        T4 tempB0 = b[pos.x + (i + 0) * size4];
        T4 tempB1 = b[pos.x + (i + 1) * size4];
        T4 tempB2 = b[pos.x + (i + 2) * size4];
        T4 tempB3 = b[pos.x + (i + 3) * size4];

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

    c[pos.x + ((pos.y << TILEY_SHIFT) + 0) * size4] = sum0;
    c[pos.x + ((pos.y << TILEY_SHIFT) + 1) * size4] = sum1;
    c[pos.x + ((pos.y << TILEY_SHIFT) + 2) * size4] = sum2;
    c[pos.x + ((pos.y << TILEY_SHIFT) + 3) * size4] = sum3;

    /*c[pos.x + ((pos.y << TILEY_SHIFT) + 0) * size4] = (float4)(0, 1, 2, 3);
    c[pos.x + ((pos.y << TILEY_SHIFT) + 1) * size4] = (float4)(4, 5, 6, 7);
    c[pos.x + ((pos.y << TILEY_SHIFT) + 2) * size4] = (float4)(8, 9, 10, 11);
    c[pos.x + ((pos.y << TILEY_SHIFT) + 3) * size4] = (float4)(12, 13, 14, 15);*/
}
