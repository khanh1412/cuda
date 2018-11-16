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

	//CLEAR BUCKETS
	for (int bucketID=0; bucketID<hm_buckets; bucketID++)
		buckets[hm_threads * bucketID + threadID] = -1;	
	__syncthreads();

	//START ROUNDS
	int value = arr[threadID];
	for (int roundID=0; roundID<hm_rounds; roundID++)
	{
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
					buckets[i] = -1;
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
void quicksort(int *arr, int size)
{
	if (size<=1) return;

	int pivot = arr[0];
	int wall = 1;
	for (int i=1; i<size; i++)
	{
		if (arr[i] < pivot)
		{
			int temp = arr[i];
			arr[i] = arr[wall];
			arr[wall] = temp;
			wall++;
		}
	}
	{
		arr[0] = arr[wall-1];
		arr[wall-1] = pivot;
	}
	quicksort(arr, wall-1);
	quicksort(arr+wall+1, size-wall-1);
}


#include<ctime>
int main()
{
	std::default_random_engine gen(std::time(nullptr));
	std::uniform_int_distribution<int> dist(0, 999);

	int size = 1000;

	int *a1 = new int[size];
	int *a2 = new int[size];
	for (int i=0; i<size; i++)
	{
		a1[i] = dist(gen);
		a2[i] = a1[i];
	}

	std::clock_t t1 = std::clock();
	radix_gpu(a1, size, 5, 6);
	std::clock_t t2 = std::clock();
	quicksort(a2, size);
	std::clock_t t3 = std::clock();
	std::printf("cpu : %d\n", t3-t2);
	std::printf("gpu : %d\n", t2-t1);
	return 0;

}
