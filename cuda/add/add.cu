#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 
// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
        // Get our global thread ID
        int id = blockIdx.x*blockDim.x + threadIdx.x;
  
        // Make sure we do not go out of bounds
        if (id < n)
            c[id] = a[id] + b[id];
}
 
int main( int argc, char* argv[] )
{
        // Size of vectors
        int n = 1000000;
  
        // Host input vectors
        double *h_a;
        double *h_b;
        //Host output vector
        double *h_c;
  
        // Device input vectors
        double *d_a;
        double *d_b;
        //Device output vector
        double *d_c;
  
        // Size, in bytes, of each vector
        size_t bytes = n*sizeof(double);
  
        // Allocate memory for each vector on host
        h_a = (double*)malloc(bytes);
        h_b = (double*)malloc(bytes);
        h_c = (double*)malloc(bytes);
  
	int i;
       	// Initialize vectors on host
        for( i = 0; i < n; i++ ) {
            h_a[i] = sin(i)*sin(i);
            h_b[i] = cos(i)*cos(i);
        }
 
 
        // Allocate memory for each vector on GPU
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);

	// Copy host vectors to device
        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
  
        int blockSize, gridSize;
  
        // Number of threads in each thread block
        blockSize = 1024;

        // Number of thread blocks in grid
        gridSize = (int)ceil((float)n/blockSize);
  
        // Execute the kernel
        vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
  
        // Copy array back to host
        cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost );
 
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

