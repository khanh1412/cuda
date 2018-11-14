NVCC=nvcc
NVCC_FLAGS= -ccbin cuda-g++ -I./include -g 

TSP:
	$(NVCC) $(NVCC_FLAGS) -o run examples/TSP.cu

sort:
	$(NVCC) $(NVCC_FLAGS) -o run examples/sort.cu
ext_sort:
	$(NVCC) $(NVCC_FLAGS) -o run examples/extended_sort.cu

