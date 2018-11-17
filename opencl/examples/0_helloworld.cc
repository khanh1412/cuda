#include<CL/cl.h>
float data[3] = {1,2,3};
unsigned int DATA_SIZE = 3;
#define LENGTH 2
int main()
{
	cl_device_id device;
	clGetDeviceIDs(nullptr, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	auto context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
	auto queue = clCreateCommandQueue(context, device, static_cast<cl_command_queue_properties>(0), nullptr);

	char *source = 
	{
		"kernel void calcSin(global float *data)\n"
		"{\n"
		"	int id = get_global_id(0);\n"
		"	data[id] = sin(data[id]);\n"
		"}\n"
	};

	auto program = clCreateProgramWithSource(context, 1, (const char**)(&source), nullptr, nullptr);
	clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
	auto kernel = clCreateKernel(program, "calcSin", nullptr);

	auto buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_SIZE, NULL, NULL);
	clEnqueueWriteBuffer(queue, buffer, CL_FALSE, 0, DATA_SIZE, data, 0, nullptr, nullptr);

	clSetKernelArg(kernel, 0, sizeof(buffer), &buffer);
	size_t global_dimensions[] = {LENGTH, 0, 0};
	clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, global_dimensions, nullptr, 0, nullptr, nullptr);
	clEnqueueReadBuffer(queue, buffer, CL_FALSE, 0, sizeof(cl_float)*LENGTH, data, 0, NULL, NULL);
	clFinish(queue);
}
