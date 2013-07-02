#define T float
#define BLOCK_SIZE 4

#define CONCAT(a, b) a ## b
#define CONCAT_EXPANED(a, b) CONCAT(a, b)


#define FACTOR(row, col, k) + aTile[row].s##k * bTile[k].s##col

#define FACTOR_2(row, col) FACTOR(row, col, 0) FACTOR(row, col, 1)
#define FACTOR_4(row, col) FACTOR(row, col, 0) FACTOR(row, col, 1) FACTOR(row, col, 2) FACTOR(row, col, 3)
#define FACTOR_8(row, col) FACTOR(row, col, 0) FACTOR(row, col, 1) FACTOR(row, col, 2) FACTOR(row, col, 3) FACTOR(row, col, 4) FACTOR(row, col, 5) FACTOR(row, col, 6) FACTOR(row, col, 7)
#define FACTOR_16(row, col) FACTOR(row, col, 0) FACTOR(row, col, 1) FACTOR(row, col, 2) FACTOR(row, col, 3) FACTOR(row, col, 4) FACTOR(row, col, 5) FACTOR(row, col, 6) FACTOR(row, col, 7) FACTOR(row, col, 8) FACTOR(row, col, 9) FACTOR(row, col, 10) FACTOR(row, col, 11) FACTOR(row, col, 12) FACTOR(row, col, 13) FACTOR(row, col, 14) FACTOR(row, col, 15)

#if BLOCK_SIZE == 2
#define FACTORS(row, col) FACTOR_2(row, col)
#elif BLOCK_SIZE == 4
#define FACTORS(row, col) FACTOR_4(row, col)
#elif BLOCK_SIZE == 8
#define FACTORS(row, col) FACTOR_8(row, col)
#elif BLOCK_SIZE == 16
#define FACTORS(row, col) FACTOR_16(row, col)
#else
#error "BLOCK_SIZE of " BLOCK_SIZE " is not supported"
#endif

#define SUM_START(row, col) sum[row].s##col = sum[row].s##col
#define SUM(row, col) SUM_START(row, col) FACTORS(row, col);

#define SUM_2(row) SUM(row, 0) SUM(row, 1)
#define SUM_4(row) SUM(row, 0) SUM(row, 1) SUM(row, 2) SUM(row, 3)
#define SUM_8(row) SUM(row, 0) SUM(row, 1) SUM(row, 2) SUM(row, 3) SUM(row, 4) SUM(row, 5) SUM(row, 6) SUM(row, 7)
#define SUM_16(row) SUM(row, 0) SUM(row, 1) SUM(row, 2) SUM(row, 3) SUM(row, 4) SUM(row, 5) SUM(row, 6) SUM(row, 7) SUM(row, 8) SUM(row, 9) SUM(row, 10) SUM(row, 11) SUM(row, 12) SUM(row, 13) SUM(row, 14) SUM(row, 15)

#if BLOCK_SIZE == 2
#define SUMS_OF_ROW(row) { SUM_2(row) }
#elif BLOCK_SIZE == 4
#define SUMS_OF_ROW(row) { SUM_4(row) }
#elif BLOCK_SIZE == 8
#define SUMS_OF_ROW(row) { SUM_8(row) }
#elif BLOCK_SIZE == 16
#define SUMS_OF_ROW(row) { SUM_16(row) }
#else
#error "BLOCK_SIZE of " BLOCK_SIZE " is not supported"
#endif

#define TB CONCAT_EXPANED(T, BLOCK_SIZE)

// Output tile size : BLOCK_SIZE x BLOCK_SIZE = Each thread computes BLOCK_SIZE x BLOCK_SIZE float values
// Required global threads = (size / BLOCK_SIZE, size / BLOCK_SIZE)
// This kernel runs on 7xx and CPU as they don't have hardware local memory 
__kernel void MultTile(__global TB* a, __global TB* b, __global TB* c, uint size)
{
    int2 pos = (int2)(get_global_id(0), get_global_id(1));

    // Vectorization of input Matrices reduces their width by a factor of BLOCK_SIZE
    uint blocks = size / BLOCK_SIZE;

    if(pos.x >= blocks || pos.y >= blocks)
        return;

    TB sum[BLOCK_SIZE];
#pragma unroll
    for(int s = 0; s < BLOCK_SIZE; s++)
        sum[s] = (TB)(0.0);

    for(int i = 0; i < size; i = i + BLOCK_SIZE)
    {
        TB aTile[BLOCK_SIZE];
        TB bTile[BLOCK_SIZE];

#pragma unroll
        for(int s = 0; s < BLOCK_SIZE; s++)
            aTile[s] = a[i / BLOCK_SIZE + ((pos.s1 * BLOCK_SIZE) + s) * blocks];

        //Matrix B is not transposed
#pragma unroll
        for(int s = 0; s < BLOCK_SIZE; s++)
            bTile[s] = b[pos.s0 + (i + s) * blocks];

#pragma unroll
        for(int row = 0; row < BLOCK_SIZE; row++)
            SUMS_OF_ROW(row)

    }

#pragma unroll
    for(int s = 0; s < BLOCK_SIZE; s++)
        c[pos.s0 + ((pos.s1 * BLOCK_SIZE) + s) * blocks] = sum[s];
}
