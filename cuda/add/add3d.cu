#include<iostream>
#include<cmath>
// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(float *a, float *b, float *c, int n)
{
	int i = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	if (i<n*n*n)
		c[i] = a[i] + b[i];
}
 
int main( int argc, char* argv[] )
{
	float *h_a = new float[100*100*100];
	float *h_b = new float[100*100*100];
	float *h_c = new float[100*100*100];

	for (int i=0; i<100*100*100; i++)
	{
		h_a[i] = std::sin(i)*std::sin(i);
		h_b[i] = std::cos(i)*std::cos(i);
	}
  
  
       // Device input vectors
        float *d_a;
        float *d_b;
        //Device output vector
        float *d_c;
  
        // Size, in bytes, of each vector
        size_t bytes = 100*100*100*sizeof(float);
 
 
        // Allocate memory for each vector on GPU
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);

	// Copy host vectors to device
        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);


	dim3 block(100, 100, 100);
	dim3 thread(1, 1, 1);


  
        // Execute the kernel
        vecAdd<<<block, thread>>>(d_a, d_b, d_c, 100);
  
        // Copy array back to host
        cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost );


	for (int i=0; i<10; i++)
		std::cout<<h_c[i];

 
        // Release device memory
        cudaFree(d_c);
        cudaFree(d_b);
        cudaFree(d_a);
  
        // Release host memory
        free(h_c);
        free(h_b);
        free(h_a);
  
        return 0;
}

