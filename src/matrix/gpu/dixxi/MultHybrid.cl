
__kernel void Mult(__read_only image2d_t a, __read_only image2d_t b, __global __write_only float* c, uint size)
{
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

    size_t col = get_global_id(0);
    size_t row = get_global_id(1);

    // check bounds
    if(col >= size || row >= size)
        return;

    float sum = 0;
    for(size_t i = 0; i < size; i++)
    {
        float4 value1 = read_imagef(a, sampler, (int2)(i, row));
        float4 value2 = read_imagef(b, sampler, (int2)(col, i));

        sum += value1.x * value2.x;
    }

    c[row * size + col] = sum;
}
