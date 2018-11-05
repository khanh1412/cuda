#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int main( int argc, char* argv[])
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
        for( i = 0; i < n; i++ ) 
	{
		h_a[i] = sin(i)*sin(i);
		h_b[i] = cos(i)*cos(i);
        }

	// CPU calculation
	
	for (i=0; i<n; i++)
	{
		h_c[i] = h_a[i] + h_b[i];
	}

        // Release host memory
        free(h_a);
        free(h_b);
        free(h_c);
}
