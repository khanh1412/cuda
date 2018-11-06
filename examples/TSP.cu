#include<random>
#include<iostream>
#include<stdio.h>
#include<ctime>

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
void matrix_randomizer(float *arr, int size)
{
	std::random_device rd;
        std::uniform_real_distribution<float> dist(0, 1);

        std::mt19937_64 prng;
        prng.seed(dist(rd));

	for (int i=0; i<size*size; i++)
	{
		arr[i] = dist(prng);
	}
}

float cpu_call(float *arr, int *perm, float *result, int size, int total_threads)
{
	auto t1 = std::clock();
	for (int id=0; id<total_threads; id++)
	{
		//permutation
		int *pos = perm + id*size;
		//traversal
		float cost = 0;
		int last, curr;
	
		for (int i=0; i<size-1; i++)
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
	auto t2 = std::clock();
	return static_cast<float>(t2-t1)/CLOCKS_PER_SEC;

}
__global__ void TSP(int *perm, float *result, float *arr, int size, int total_threads)
{
	int block_id = blockIdx.x;
	int thread_id = threadIdx.x;
	int threads = blockDim.x;
	int id = thread_id + block_id * threads;
	if (id >= total_threads) return;
	//permutation
	int *pos = perm + id*size;
	//traversal
	float cost = 0;
	int last, curr;
	
	for (int i=0; i<size-1; i++)
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
float gpu_call(float *arr, int *perm, float *result, int size, int total_threads)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int MaxThreadsPerBlock = prop.maxThreadsPerBlock;
	int HmBlocks = 1 + total_threads/MaxThreadsPerBlock;

	dim3 BlocksPerGrid(HmBlocks, 1, 1);
	dim3 ThreadsPerBlock(MaxThreadsPerBlock,1,1);

	float *d_arr; cudaMalloc(&d_arr, size*size*sizeof(float));
	int *d_perm; cudaMalloc(&d_perm, total_threads*size*sizeof(float));
	float *d_result; cudaMalloc(&d_result, total_threads*sizeof(float));


	cudaMemcpy(d_perm, perm, total_threads*size*sizeof(float), cudaMemcpyHostToDevice);
	auto t1 = std::clock();
	cudaMemcpy(d_arr, arr, size*size*sizeof(float), cudaMemcpyHostToDevice);
		TSP<<<BlocksPerGrid, ThreadsPerBlock>>>(d_perm, d_result, d_arr, size, total_threads);
	cudaMemcpy(result, d_result, total_threads*sizeof(float), cudaMemcpyDeviceToHost);
	auto t2 = std::clock();

	cudaFree(d_arr);
	cudaFree(d_perm);
	cudaFree(d_result);

	return static_cast<float>(t2-t1)/CLOCKS_PER_SEC;
}


int main(int argc, char *argv[])
{
	int size;
	if (argc == 1)
		size = 2;
	else
		size = std::stoi(argv[1]);

	float *arr = new float[size*size];
	matrix_randomizer(arr, size);

	int total_threads = 1;
	for (int i=2; i<=size; i++)
		total_threads *= i;

	perm = new int[total_threads*size];

	int *a = new int[size];
	for (int i=0; i<size; i++)
		a[i] = i;

	auto t1 = std::clock();
	std::cout<<"permutation started!"<<std::endl;
	heapPermutation(a, size, size);
	std::cout<<"permutation done!"<<std::endl;
	auto t2 = std::clock();
	std::cout<<"permutation time: "<<1000000*static_cast<float>(t2-t1)/CLOCKS_PER_SEC<<" (microsec)"<<std::endl;

	delete a;

	float *result = new float[total_threads];

	auto cpu_time = cpu_call(arr, perm, result, size, total_threads);
	auto gpu_time = gpu_call(arr, perm, result, size, total_threads);
	
		
	std::cout<<"cpu time = "<<1000000*cpu_time<<" (microsec)"<<std::endl;
	std::cout<<"gpu time = "<<1000000*gpu_time<<" (mircosec)"<<std::endl;

	delete perm;
	delete arr;

}	
