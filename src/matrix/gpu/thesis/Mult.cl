__kernel void NaiveGPU(__global float* a, __global float* b, __global float* c, uint n) {
	uint col = get_global_id(0);
	uint row = get_global_id(1);

	if (row >= n || col >= n)
		return;

	float sum = 0.0f; 
	for (uint i = 0; i < n; i++)
		sum += a[row * n + i] * b[i * n + col];

	c[row * n + col] = sum;
} // NaiveGPU
