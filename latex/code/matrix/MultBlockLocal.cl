#define TILE_SIZE 16
#define BLOCK_SIZE 4

__kernel void MultBlockLocal(__global float4* a, __global float4* b, __global float4* c, uint size)
{
  int tilePos = get_local_id(0) + (get_local_id(1) * BLOCK_SIZE) * TILE_SIZE;
  int globalPosC = (get_global_id(1) * BLOCK_SIZE) * get_global_size(0) + get_global_id(0);

  float4 sum0 = (float4)(0);
  float4 sum1 = (float4)(0);
  float4 sum2 = (float4)(0);
  float4 sum3 = (float4)(0);

  int size4 = size / BLOCK_SIZE;

  // This loop runs for the number of tiles
  for(int tileIndex = 0; tileIndex < (size4 / TILE_SIZE); tileIndex++)
  {
    __local float4 aTile[TILE_SIZE * TILE_SIZE * BLOCK_SIZE];
    __local float4 bTile[TILE_SIZE * TILE_SIZE * BLOCK_SIZE];

    int globalPosA = (get_global_id(1) * BLOCK_SIZE) * get_global_size(0) + tileIndex * TILE_SIZE + get_local_id(0);
    int globalPosB = ((tileIndex * TILE_SIZE + get_local_id(1)) * BLOCK_SIZE) * get_global_size(0) + get_global_id(0);
		
    // Load values in aTile from matrix a
    aTile[tilePos + 0 * TILE_SIZE] = a[globalPosA + 0 * size4];
    aTile[tilePos + 1 * TILE_SIZE] = a[globalPosA + 1 * size4];
    aTile[tilePos + 2 * TILE_SIZE] = a[globalPosA + 2 * size4];
    aTile[tilePos + 3 * TILE_SIZE] = a[globalPosA + 3 * size4];

    // Load values in bTile from matrix b
    bTile[tilePos + 0 * TILE_SIZE] = b[globalPosB + 0 * size4];
    bTile[tilePos + 1 * TILE_SIZE] = b[globalPosB + 1 * size4];
    bTile[tilePos + 2 * TILE_SIZE] = b[globalPosB + 2 * size4];
    bTile[tilePos + 3 * TILE_SIZE] = b[globalPosB + 3 * size4];

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int j = 0; j < TILE_SIZE; j++)
    {
      // Load 4 float4s from aTile
      float4 aBlock0 = aTile[j + (get_local_id(1) * BLOCK_SIZE + 0) * TILE_SIZE];
      float4 aBlock1 = aTile[j + (get_local_id(1) * BLOCK_SIZE + 1) * TILE_SIZE];
      float4 aBlock2 = aTile[j + (get_local_id(1) * BLOCK_SIZE + 2) * TILE_SIZE];
      float4 aBlock3 = aTile[j + (get_local_id(1) * BLOCK_SIZE + 3) * TILE_SIZE];

      // Load 4 float4s from bTile
      float4 bBlock0 = bTile[get_local_id(0) + (j * BLOCK_SIZE + 0) * TILE_SIZE];
      float4 bBlock1 = bTile[get_local_id(0) + (j * BLOCK_SIZE + 1) * TILE_SIZE];
      float4 bBlock2 = bTile[get_local_id(0) + (j * BLOCK_SIZE + 2) * TILE_SIZE];
      float4 bBlock3 = bTile[get_local_id(0) + (j * BLOCK_SIZE + 3) * TILE_SIZE];

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
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  c[globalPosC + 0 * get_global_size(0)] = sum0;
  c[globalPosC + 1 * get_global_size(0)] = sum1;
  c[globalPosC + 2 * get_global_size(0)] = sum2;
  c[globalPosC + 3 * get_global_size(0)] = sum3;
}
