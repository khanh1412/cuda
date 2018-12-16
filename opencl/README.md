# setup OpenCL

NVIDIA : headers :  `/usr/local/cuda/include` lib : `/usr/local/cuda/lib64`

INTEL : `sudo dnf install mesa-libOpenCL-devel opencl-headers`

AMD : `no idea`

	For my AMD card, using the mesa-libOpenCL. It works for some simple tasks but unfortunately, when I tried to do the asynchronous copy data with AMD device platform, it seems not to asynchronously run.
	
