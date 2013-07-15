#define TILE_SIZE 16

__kernel void MultLocal(__global float* a, __global float* b, __global float* c, uint size)
{
  int row = get_global_id(0);
  int col = get_global_id(1);
  int tileX = get_group_id(0);
  int tileY = get_group_id(1);
  int localX = get_local_id(0);
  int localY = get_local_id(1);

  int aPos = size * (TILE_SIZE * tileY);
  int bPos = TILE_SIZE * tileX;
  int aEnd = aPos + size;

  float sum = 0.0f;

  while(aPos < aEnd)
  {
    __local float aTile[TILE_SIZE][TILE_SIZE];
    __local float bTile[TILE_SIZE][TILE_SIZE];

    aTile[localY][localX] = a[size * localY + localX + aPos];
    bTile[localY][localX] = b[size * localY + localX + bPos];

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int k = 0; k < TILE_SIZE; k++)
      sum += aTile[localY][k] * bTile[k][localX];

    barrier(CLK_LOCAL_MEM_FENCE);

    aPos += TILE_SIZE;
    bPos += TILE_SIZE * size;
  }

  c[col * size + row] = sum;
}
