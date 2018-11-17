CC=g++
CC_FLAGS= -std=c++17 -g 
INCLUDE= -I/usr/local/cuda/include
LIBRARY= -L/usr/local/cuda/lib64 -lOpenCL 

helloworld:
	$(CC) $(CC_FLAGS) $(INCLUDE) -o run examples/0_helloworld.cc $(LIBRARY)
