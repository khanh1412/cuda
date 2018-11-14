#include<random>
#include<ctime>
#include<iostream>


__device__ int getBucketID(int value, int hm_buckets_per_block, int roundID)
{
	int bucketID = value;
	for (int i=0; i<roundID; i++)
		bucketID /= hm_buckets_per_block;
	return bucketID % hm_buckets_per_block;
}

__global__ void radix(int *arr, int *buckets, int hm_buckets_per_block, int hm_rounds, int size)
{
	int threadID = threadIdx.x;
	int blockID = blockIdx.x;
	int hm_threads_per_block = blockDim.x;

	int buckets_offset = hm_threads_per_block * hm_buckets_per_block * blockID;


	int hm_threads = hm_threads_per_block;
	if (blockID + 1 == gridDim.x) //last block
		hm_threads = size % hm_threads_per_block;

	//START ROUNDS
	int value = arr[threadID];
	for (int roundID=0; roundID<hm_rounds; roundID++)
	{
		//CLEAR BUCKETS
		for (int bucketID=0; bucketID<hm_buckets_per_block; bucketID++)
			buckets[buckets_offset + hm_threads * bucketID + threadID] = -1;	
		__syncthreads();
		//Find the bucketID
		int bucketID = getBucketID(value, hm_buckets_per_block, roundID);
		//Fill in the buckets
		buckets[ buckets_offset + hm_threads * bucketID + threadID] = value;
		__syncthreads();
		//Fill out
		int counter = 0;
		for (int i=0; i<hm_threads*hm_buckets_per_block; i++)
		{
			if (buckets[buckets_offset + i] > -0.5)
				if (counter == threadID)
				{
					value = buckets[buckets_offset + i];
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
   hm_buckets_per_block^hm_rounds > maximum value in array.
 */
#include<vector>
int getMax(std::vector<std::vector<int>*> *set)
{
	int max_vec = 0;
	for (int i=0; i<set->size(); i++)
	{
		if (not set->at(i)->empty())
			max_vec = i;
	}
	for (int i=0; i<set->size(); i++)
	{
		if ((not set->at(i)->empty()) and set->at(i)->back() > set->at(max_vec)->back())
			max_vec = i;
	}
	int value = set->at(max_vec)->back();
	set->at(max_vec)->pop_back();
	return value;
}

void ext_radix_gpu(int *arr, int size, int hm_buckets_per_block, int hm_rounds)
{
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        int hm_threads_per_block = prop.maxThreadsPerBlock;

	int hm_blocks = size / hm_threads_per_block;

	int *buckets; cudaMalloc(&buckets, hm_blocks * hm_buckets_per_block * hm_threads_per_block * sizeof(int));

	dim3 BlocksPerGrid(hm_blocks, 1, 1);
	dim3 ThreadsPerBlock(hm_threads_per_block, 1, 1);

	int *d_arr; cudaMalloc(&d_arr, size * sizeof(int));
	cudaMemcpy(d_arr, arr, size*sizeof(int), cudaMemcpyHostToDevice);
	radix<<<BlocksPerGrid, ThreadsPerBlock>>>(d_arr, buckets, hm_buckets_per_block, hm_rounds, size);
	cudaMemcpy(arr, d_arr, size*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(buckets);

	std::printf("gpu sort done!\n");
	if (hm_blocks > 1)//merge
	{
		int i=0;

		auto set = new std::vector<std::vector<int>*>();
		while (i<size)
		{
			if (i % hm_threads_per_block == 0)
				set->push_back(new std::vector<int>());
			set->back()->push_back(arr[i]);
			i++;
		}
		while (i<size)
		{
			arr[i] = getMax(set);
		}
	}
}


int main()
{
	//std::default_random_engine gen(std::time(nullptr));
	//std::uniform_int_distribution<int> dist(0, 999);

	int *arr = new int[5000];
	for (int i=0; i<5000; i++)
	{
		//arr[i] = dist(gen);
		arr[i] = 5000-i;
	}


	ext_radix_gpu(arr, 5000, 10, 4);

	for (int i=0; i<5000; i++)
		std::printf("%d ", arr[i]);

	return 0;

}
