all: cpu gpu

cpu:
	g++ -o cpu add.c
gpu:
	nvcc -ccbin cuda-g++ -o gpu add.cu
