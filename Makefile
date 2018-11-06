NVCC=nvcc
NVCC_FLAGS= -ccbin cuda-g++ -I./include

all:
	$(NVCC) $(NVCC_FLAGS) -o run examples/TSP.cu

