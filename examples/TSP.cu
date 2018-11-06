#include<random>
#include<iostream>

__global__ void TSP(int *perm, float *result, float *d_arr, int size, int total_threads)
{
	int block_id = blockIdx.x;
	int thread_id = threadIdx.x;
	int threads = blockDim.x;
	int id = thread_id + block_id * threads;
	if (id >= total_threads) return;
	//copy data to shared memory
	extern __shared__ float arr[];
	for (int i=thread_id; i<total_threads; i+= threads)
	{
		arr[i] = d_arr[i];
	}

	__syncthreads();
	//permutation
	int *pos = perm + id*size;
	float cost = 0;
	int last, curr;
	
	for (int i=0; i<size; i++)
	{
		last = pos[i];
		curr = pos[i+1];
		cost += arr[last*size + curr];
	}
	last = pos[size-1];
	curr = pos[0];
	cost += arr[last*size + curr];

	result[id] = cost;
}

int *perm;
int count = 0;

void print(int a[], int n)
{
	for (int i=0; i<n; i++)
	{
		perm[count] = a[i];
		count++;
		std::cout<<a[i]<<" ";
	}
	std::cout<<std::endl;
}

void heapPermutation(int a[], int size, int n)
{
	if (size==1)
	{
		print(a, n);
		return;
	}
	else
	{
		for (int i=0; i<size; i++)
		{
			heapPermutation(a, size-1, n);
			if (size%2==1)
				std::swap(a[0], a[size-1]);
			else
				std::swap(a[i], a[size-1]);
		}
	}
}
void matrix_randomizer(float density, float *arr, int size)
{
	std::random_device rd;
        std::uniform_real_distribution<float> dist(0, 1);

        std::mt19937_64 prng;
        prng.seed(dist(rd));

	for (int i=0; i<size*size; i++)
	{
		arr[i] = 0.0;
		if (dist(prng) < density)
		{
			arr[i] = dist(prng);
		}
	}
}
int main()
{
	int size = 10;
	float density = 0.5;

	float *h_arr = new float[size*size];
	matrix_randomizer(density, h_arr, size);

	int threads = 1;
	for (int i=2; i<=size; i++)
		threads *= i;

	perm = new int[threads*size];

	int *a = new int[size];
	for (int i=0; i<size; i++)
		a[i] = i;
	std::cout<<"permutation started!"<<std::endl;
	heapPermutation(a, size, size);
	std::cout<<"permutation done!"<<std::endl;

	size_t bytes = size*size*sizeof(float);
	float *d_arr; cudaMalloc(&d_arr, bytes);
	int *d_perm; cudaMalloc(&d_perm, threads*size*sizeof(float));
	float *d_result; cudaMalloc(&d_result, threads*sizeof(float));

	cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_perm, perm, threads*size*sizeof(float), cudaMemcpyHostToDevice);



	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int MaxThreadsPerBlock = prop.maxThreadsPerBlock;

	int HmBlocks = 1 + threads/MaxThreadsPerBlock;


	dim3 BlocksPerGrid(HmBlocks, 1, 1);
	dim3 ThreadsPerBlock(MaxThreadsPerBlock,1,1);

	TSP<<<BlocksPerGrid, ThreadsPerBlock, bytes>>>(d_perm, d_result, d_arr, size, threads);


	float *result = new float[threads];

	cudaMemcpy(result, d_result, threads*sizeof(float), cudaMemcpyDeviceToHost);
	
	count = 0;
	for (int i=0; i<threads; i++)
	{
		std::cout<<result[i]<<" : [";
		for (int j=0; j<size; j++)
		{
			std::cout<<perm[count]<<" ";
			count++;
		}
		std::cout<<"]"<<std::endl;
	}




}	