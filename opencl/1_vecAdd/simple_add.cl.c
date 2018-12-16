void kernel simple_add(global const float* A, 
		       global const float* B, 
		       global float* C)
{
	int i = get_global_id(0);
	C[i] = A[i] + B[i];
}
