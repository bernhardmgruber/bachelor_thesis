#define TILE_SIZE 16
#define BLOCK_SIZE 4

__kernel void BlocksAndTilesGPU(__global float4* a, __global float4* b, __global float4* c, uint n) {
	uint col     = get_global_id(0);
	uint row     = get_global_id(1);
	uint localX  = get_local_id(0);
	uint localY  = get_local_id(1);
	uint n4      = n / BLOCK_SIZE;

	uint posA    = (row * BLOCK_SIZE) * n4 + localX;
	uint posB    = (localY * BLOCK_SIZE) * n4 + col;
	uint stepA   = TILE_SIZE;
	uint stepB   = TILE_SIZE * BLOCK_SIZE * n4;
	uint endA    = posA + n4;
	uint tilePos = localX + (localY * BLOCK_SIZE) * TILE_SIZE;

	float4 sum0  = (float4)(0.0f);
	float4 sum1  = (float4)(0.0f);
	float4 sum2  = (float4)(0.0f);
	float4 sum3  = (float4)(0.0f);

	while (posA < endA) {
		__local float4 aTile[TILE_SIZE * TILE_SIZE * BLOCK_SIZE];
		__local float4 bTile[TILE_SIZE * TILE_SIZE * BLOCK_SIZE];

		aTile[tilePos + 0 * TILE_SIZE] = a[posA + 0 * n4];
		aTile[tilePos + 1 * TILE_SIZE] = a[posA + 1 * n4];
		aTile[tilePos + 2 * TILE_SIZE] = a[posA + 2 * n4];
		aTile[tilePos + 3 * TILE_SIZE] = a[posA + 3 * n4];
		bTile[tilePos + 0 * TILE_SIZE] = b[posB + 0 * n4];
		bTile[tilePos + 1 * TILE_SIZE] = b[posB + 1 * n4];
		bTile[tilePos + 2 * TILE_SIZE] = b[posB + 2 * n4];
		bTile[tilePos + 3 * TILE_SIZE] = b[posB + 3 * n4];

		barrier(CLK_LOCAL_MEM_FENCE);

		for (uint k = 0; k < TILE_SIZE; k++) {
			float4 blA0 = aTile[k + (localY * BLOCK_SIZE + 0) * TILE_SIZE];
			float4 blA1 = aTile[k + (localY * BLOCK_SIZE + 1) * TILE_SIZE];
			float4 blA2 = aTile[k + (localY * BLOCK_SIZE + 2) * TILE_SIZE];
			float4 blA3 = aTile[k + (localY * BLOCK_SIZE + 3) * TILE_SIZE];
			float4 blB0 = bTile[localX + (k * BLOCK_SIZE + 0) * TILE_SIZE];
			float4 blB1 = bTile[localX + (k * BLOCK_SIZE + 1) * TILE_SIZE];
			float4 blB2 = bTile[localX + (k * BLOCK_SIZE + 2) * TILE_SIZE];
			float4 blB3 = bTile[localX + (k * BLOCK_SIZE + 3) * TILE_SIZE];

			sum0.x += blA0.x * blB0.x + blA0.y * blB1.x + blA0.z * blB2.x + blA0.w * blB3.x;
			sum0.y += blA0.x * blB0.y + blA0.y * blB1.y + blA0.z * blB2.y + blA0.w * blB3.y;
			sum0.z += blA0.x * blB0.z + blA0.y * blB1.z + blA0.z * blB2.z + blA0.w * blB3.z;
			sum0.w += blA0.x * blB0.w + blA0.y * blB1.w + blA0.z * blB2.w + blA0.w * blB3.w;
			sum1.x += blA1.x * blB0.x + blA1.y * blB1.x + blA1.z * blB2.x + blA1.w * blB3.x;
			sum1.y += blA1.x * blB0.y + blA1.y * blB1.y + blA1.z * blB2.y + blA1.w * blB3.y;
			sum1.z += blA1.x * blB0.z + blA1.y * blB1.z + blA1.z * blB2.z + blA1.w * blB3.z;
			sum1.w += blA1.x * blB0.w + blA1.y * blB1.w + blA1.z * blB2.w + blA1.w * blB3.w;
			sum2.x += blA2.x * blB0.x + blA2.y * blB1.x + blA2.z * blB2.x + blA2.w * blB3.x;
			sum2.y += blA2.x * blB0.y + blA2.y * blB1.y + blA2.z * blB2.y + blA2.w * blB3.y;
			sum2.z += blA2.x * blB0.z + blA2.y * blB1.z + blA2.z * blB2.z + blA2.w * blB3.z;
			sum2.w += blA2.x * blB0.w + blA2.y * blB1.w + blA2.z * blB2.w + blA2.w * blB3.w;
			sum3.x += blA3.x * blB0.x + blA3.y * blB1.x + blA3.z * blB2.x + blA3.w * blB3.x;
			sum3.y += blA3.x * blB0.y + blA3.y * blB1.y + blA3.z * blB2.y + blA3.w * blB3.y;
			sum3.z += blA3.x * blB0.z + blA3.y * blB1.z + blA3.z * blB2.z + blA3.w * blB3.z;
			sum3.w += blA3.x * blB0.w + blA3.y * blB1.w + blA3.z * blB2.w + blA3.w * blB3.w;
		} // for
		barrier(CLK_LOCAL_MEM_FENCE);

		posA += stepA;
		posB += stepB;
	} // while

	uint posC = (row * BLOCK_SIZE) * n4 + col;
	c[posC + 0 * n4] = sum0;
	c[posC + 1 * n4] = sum1;
	c[posC + 2 * n4] = sum2;
	c[posC + 3 * n4] = sum3;
} // BlocksAndTilesGPU
