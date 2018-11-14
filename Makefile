NVCC=nvcc
NVCC_FLAGS= -ccbin cuda-g++ -I./include

TSP:
	$(NVCC) $(NVCC_FLAGS) -o run examples/TSP.cu

sort:
	$(NVCC) $(NVCC_FLAGS) -o run examples/sort.cu

