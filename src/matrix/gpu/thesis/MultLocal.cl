#define TILE_n 16

__kernel void MultLocal(__global float* a, __global float* b, __global float* c, uint n) {
	uint col = get_global_id(0);
	uint row = get_global_id(1);
	uint localX = get_local_id(0);
	uint localY = get_local_id(1);

	uint posA = row * n + localX;
	uint posB = localY * n + col;
	uint stepA = TILE_n;
	uint stepB = TILE_n * n;
	uint endA = posA + n;
	uint tilePos = localY * TILE_n + localX;

	float sum = 0.0f;

	while(posA < endA) {
		__local float tileA[TILE_n * TILE_n];
		__local float tileB[TILE_n * TILE_n];

		tileA[tilePos] = a[posA];
		tileB[tilePos] = b[posB];
		barrier(CLK_LOCAL_MEM_FENCE);

		for(uint k = 0; k < TILE_n; k++)
			sum += tileA[localY * TILE_n + k] * tileB[k * TILE_n + localX];
		barrier(CLK_LOCAL_MEM_FENCE);

		posA += stepA;
		posB += stepB;
	}

	c[row * n + col] = sum;
}
