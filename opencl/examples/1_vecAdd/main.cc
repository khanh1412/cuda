#include<CL/cl.hpp>
#include<iostream>
#include<stdexcept>
#include<fstream>
#include<string>

std::string readKernel(const std::string& filepath)
{
	std::ifstream ifs(filepath);
	std::string content((std::istreambuf_iterator<char>(ifs)), 
			(std::istreambuf_iterator<char>()));
	return content;
}







int main()
{
	//get all platforms (driver)
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if (0 == all_platforms.size())
		throw std::runtime_error("No platforms found. Check OpenCL installation!");

	cl::Platform default_platform = all_platforms[0];
	std::cout <<"Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<std::endl;

	//get default device of the default platform
	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);

	if (0 == all_devices.size())
		throw std::runtime_error("No devices found. Check OpenCL installation!");

	cl::Device default_device = all_devices[0];
	std::cout<<"Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<std::endl;

	cl::Context context({default_device});

	cl::Program::Sources sources;

	// kernel calculates for each element C=A+B
	std::string kernel_code = readKernel("kernel.cl");

	sources.push_back({kernel_code.c_str(), kernel_code.length()});

	cl::Program program(context, sources);

	if (CL_SUCCESS != program.build({default_device}))
		throw std::runtime_error("Error building: " + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device));
	
	cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int)*10);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int)*10);
	cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int)*10);

	int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	int B[] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

	cl::CommandQueue queue(context, default_device);

	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int)*10, A);
	queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int)*10, B);

	cl::Kernel simple_add(program, "simple_add");
	simple_add.setArg(0, buffer_A);
	simple_add.setArg(1, buffer_B);
	simple_add.setArg(2, buffer_C);

	queue.enqueueNDRangeKernel(simple_add, cl::NullRange, cl::NDRange(10, 1), cl::NullRange, nullptr, nullptr);

	

	int C[10];
	//read result C from the device to array C
	queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int)*10, C);

	std::cout<<" result: \n";
	for(int i=0; i<10; i++)
	{
		std::cout<<C[i]<<" ";
	}
	return 0;
}
