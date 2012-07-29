
__kernel void ParallelSelectionSort(__global const int* in,__global int* out)
{
    int id = get_global_id(0);
    int n = get_global_size(0);

    int element = in[id];

    int pos = 0;
    for (int i=0; i<n; i++)
    {
        int precedingElement = in[i];
        bool smaller = (precedingElement < element) || (precedingElement == element && i < id);
        pos += smaller ? 1 : 0;
    }

    out[pos] = element;
}
