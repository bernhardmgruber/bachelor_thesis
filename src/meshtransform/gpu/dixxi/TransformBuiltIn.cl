
__kernel void TransformBuiltIn(__constant T* matrix, __global T* vertices)
{
    size_t id = get_global_id(0);

    float4 m0;
    m0.s0 = matrix[0];
    m0.s1 = matrix[1];
    m0.s2 = matrix[2];
    m0.s3 = matrix[3];

    float4 m1;
    m1.s0 = matrix[4];
    m1.s1 = matrix[5];
    m1.s2 = matrix[6];
    m1.s3 = matrix[7];

    float4 m2;
    m2.s0 = matrix[8];
    m2.s1 = matrix[9];
    m2.s2 = matrix[10];
    m2.s3 = matrix[11];

//    float4 m3;
//    m3.s0 = matrix[12];
//    m3.s1 = matrix[13];
//    m3.s2 = matrix[14];
//    m3.s3 = matrix[15];

    float4 v[BLOCK_SIZE];

    // READ
    #pragma unroll BLOCK_SIZE
    for(int i = 0; i < BLOCK_SIZE; i++)
    {
        v[i].s0 = vertices[id * 3 + 0];
        v[i].s1 = vertices[id * 3 + 1];
        v[i].s2 = vertices[id * 3 + 2];
        v[i].s3 = 1;
    }

    // CALC
    float4 r[BLOCK_SIZE];

    #pragma unroll BLOCK_SIZE
    for(int i = 0; i < BLOCK_SIZE; i++)
    {
        r[i].s0 = dot(m0, v[i]);
        r[i].s1 = dot(m1, v[i]);
        r[i].s2 = dot(m2, v[i]);
    }

    // WRITE
    #pragma unroll BLOCK_SIZE
    for(int i = 0; i < BLOCK_SIZE; i++)
    {
        vertices[id * 3 + 0] = r[i].s0;
        vertices[id * 3 + 1] = r[i].s1;
        vertices[id * 3 + 2] = r[i].s2;
    }
}

