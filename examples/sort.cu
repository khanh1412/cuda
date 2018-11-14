#include<random>
#include<ctime>
#include<iostream>


__device__ int getBucketID(int value, int hm_buckets, int roundID)
{
	int bucketID = value;
	for (int i=0; i<roundID; i++)
		bucketID /= hm_buckets;
	return bucketID % hm_buckets;
}

__device__ void printBuckets(int *buckets, int hm_buckets, int hm_threads)
{
	for (int bucketID=0; bucketID < hm_buckets; bucketID++)
	{
	std::printf("Bucket %d: ", bucketID);
	for (int threadID=0; threadID < hm_threads; threadID++)
	{
		int i = threadID + bucketID * hm_threads;
		std::printf("%d ", buckets[i]);
	}
	std::printf("\n");
	}
	std::printf("\n");

}
__global__ void radix(int *arr, int *buckets, int hm_buckets, int hm_rounds, int hm_threads)
{
	int threadID = threadIdx.x;


	//START ROUNDS
	int value = arr[threadID];
	for (int roundID=0; roundID<hm_rounds; roundID++)
	{
		//CLEAR BUCKETS
		for (int bucketID=0; bucketID<hm_buckets; bucketID++)
			buckets[hm_threads * bucketID + threadID] = -1;	
		__syncthreads();
		//Find the bucketID
		int bucketID = getBucketID(value, hm_buckets, roundID);
		//Fill in the buckets
		buckets[hm_threads * bucketID + threadID] = value;
		__syncthreads();
		//Fill out
		int counter = 0;
		for (int i=0; i<hm_threads*hm_buckets; i++)
		{
			if (buckets[i] > -0.5)
				if (counter == threadID)
				{
					value = buckets[i];
					break;
				}
				else
					counter++;
		}
		__syncthreads();
	}
	//END ROUNDS
	arr[threadID] = value;
}


/* RADIX SORT
   hm_buckets^hm_rounds > maximum value in array.
 */
void radix_gpu(int *arr, int size, int hm_buckets, int hm_rounds)
{
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        int MaxThreadsPerBlock = prop.maxThreadsPerBlock;
	if (size > MaxThreadsPerBlock)
		__builtin_trap();




	int *buckets; cudaMalloc(&buckets, hm_buckets * size * sizeof(int));

	dim3 BlocksPerGrid(1, 1, 1);
	dim3 ThreadsPerBlock(size, 1, 1);

	int *d_arr; cudaMalloc(&d_arr, size * sizeof(int));
	cudaMemcpy(d_arr, arr, size*sizeof(int), cudaMemcpyHostToDevice);
	radix<<<BlocksPerGrid, ThreadsPerBlock>>>(d_arr, buckets, hm_buckets, hm_rounds, size);
	cudaMemcpy(arr, d_arr, size*sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(buckets);
}


int main()
{
	//std::default_random_engine gen(std::time(nullptr));
	//std::uniform_int_distribution<int> dist(0, 999);

	int *arr = new int[50];
	for (int i=0; i<50; i++)
	{
		//arr[i] = dist(gen);
		arr[i] = 50-i;
	}


	radix_gpu(arr, 50, 10, 3);

	for (int i=0; i<50; i++)
		std::printf("%d ", arr[i]);

	return 0;

}
