NVCC=nvcc
NVCC_FLAGS= -ccbin cuda-g++ -I./include -g 

TSP:
	$(NVCC) $(NVCC_FLAGS) -o run TSP.cu

sort:
	$(NVCC) $(NVCC_FLAGS) -o run sort.cu
ext_sort:
	$(NVCC) $(NVCC_FLAGS) -o run extended_sort.cu

