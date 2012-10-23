
inline void matrixMultiplication(__constant T* m, __global T* v, T* r)
{
    r[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3];
    r[1] = m[4] * v[0] + m[5] * v[1] + m[6] * v[2] + m[7];
    r[2] = m[8] * v[0] + m[9] * v[1] + m[10] * v[2] + m[11];
}

__kernel void Transform(__constant T* matrix, __global T* vertices)
{
    size_t id = get_global_id(0);

    float r[3];
    matrixMultiplication(matrix, &vertices[id * 3], &r);

    vertices[id * 3 + 0] = r[0];
    vertices[id * 3 + 1] = r[1];
    vertices[id * 3 + 2] = r[2];
}

