#include<CL/cl.hpp>
#include<iostream>
#include<stdexcept>
#include<fstream>
#include<string>

#include<ctime>

std::string readKernel(const std::string& filepath)
{
	std::ifstream ifs(filepath);
	std::string content((std::istreambuf_iterator<char>(ifs)), 
			std::istreambuf_iterator<char>());
	return content;
}

void clAdd(float *C, float *A, float *B, uint64_t COUNT)
{
	//PLATFORMS (OPENCL SOFTWARE)
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if (0 == all_platforms.size())
		throw std::runtime_error("No platforms found. Check OpenCL installation!");

	std::cout<<"All platforms:"<<std::endl;
	for (auto& platform : all_platforms)
		std::cout<<"\t"<<platform.getInfo<CL_PLATFORM_NAME>()<<std::endl;

	cl::Platform default_platform = all_platforms[0];
	std::cout <<"Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<std::endl;

	//DEVICES (HARDWARE)
	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);

	if (0 == all_devices.size())
		throw std::runtime_error("No devices found. Check OpenCL installation!");

	std::cout<<"Platform's devices:"<<std::endl;
	for (auto& device : all_devices)
		std::cout<<"\t"<<device.getInfo<CL_DEVICE_NAME>()<<std::endl;

	cl::Device default_device = all_devices[0];
	std::cout<<"Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<std::endl;

	//CONTEXT
	cl::Context context({default_device});

	//BUFFER
	cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, COUNT*sizeof(float));
	cl::Buffer buffer_B(context, CL_MEM_READ_ONLY, COUNT*sizeof(float));
	cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, COUNT*sizeof(float));

	//CREATE QUEUE
	cl::CommandQueue queue(context, default_device);

	//COMPILE PROGRAM
	auto t1 = std::clock();
	cl::Program::Sources sources;

	std::string kernel_code = readKernel("simple_add.cl.c");
	sources.push_back({kernel_code.c_str(), kernel_code.length()});

	cl::Program program(context, sources);

	if (CL_SUCCESS != program.build({default_device}))
		throw std::runtime_error("Error building: " + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device));
	

	cl::Kernel simple_add(program, "simple_add");
	simple_add.setArg(0, buffer_A);
	simple_add.setArg(1, buffer_B);
	simple_add.setArg(2, buffer_C);


	//COPY HOST TO DEVICE

	queue.enqueueWriteBuffer(buffer_A, CL_FALSE, 0, COUNT*sizeof(float), A);
	queue.enqueueWriteBuffer(buffer_B, CL_FALSE, 0, COUNT*sizeof(float), B);
	//CALL THE KERNEL
	queue.enqueueNDRangeKernel(simple_add, cl::NullRange, cl::NDRange(COUNT, 1, 1), cl::NDRange(1, 1, 1), nullptr, nullptr);
	//COPY DEVICE TO HOST
	queue.enqueueReadBuffer(buffer_C, CL_FALSE, 0, COUNT*sizeof(float), C);

	//queue.enqueueBarrierWithWaitList();
	queue.finish();
	auto t2 = std::clock();
	std::cout<<"CL time: "<<t2-t1<<" ticks"<<std::endl;

}


void nativeAdd(float *C, float *A, float *B, uint64_t COUNT)
{
	auto t1 = std::clock();
	for (uint64_t i = 0; i < COUNT; i++)
		C[i] = A[i] + B[i];
	auto t2 = std::clock();
	std::cout<<"Native time: "<<t2-t1<<" ticks"<<std::endl;

}



const uint64_t COUNT = 100000000;

int main()
{
	//generate data
	auto *A = new float[COUNT];
	auto *B = new float[COUNT];
	auto *C = new float[COUNT];
	for (int i=0; i<COUNT; i++)
	{
		A[i] = i + 1;
		B[i] = COUNT - i;
	}

	clAdd(C, A, B, COUNT);
	
	delete A; delete B; delete C;
	return 0;
}

