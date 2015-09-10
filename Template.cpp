#include <CL/cl.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <string>
#include <fstream>
#include <iostream>

#define ENABLE_DEBUG_PRINT

#define NON_BLOCKING_READ_WRITE 0
#define BLOCKING_READ_WRITE 1
#define KERNEL_LEAF_WG_SZ 256

cl_context          context;
cl_kernel			kernel_leaf;              // compute kernel_leaf
cl_kernel		    kernel_merge;             // compute kernel_merge
cl_command_queue	commands;				  // compute command queue

template <class T>
class MergeSortData {
public:
	T* elems;     // elements
	T* elems_op_cpu;
	T* elems_op_gpu;
	int length;
	cl_mem cl_array1, cl_array2;
	cl_event kernel_event;

	MergeSortData(int);
	void Merge(T[], int , int, int);  
	void MergeSort(T[], int, int);
	void Get_GPU_op(void);
	void Verify_GPU_op(void);
};



template <class T>
MergeSortData<T>::MergeSortData(int len)
{
	int i, status;
	/*constructor for initializing the input array*/
	length = len;
	elems = new T[length];
	elems_op_gpu = new T[length];
#ifdef	ENABLE_DEBUG_PRINT
	std::cout << "The input array is:\n";
#endif	
	for (i = 0; i < length; i++)
	{
		elems[i] = rand() % 10000;
#ifdef	ENABLE_DEBUG_PRINT
		std::cout << elems[i]<<"\t";
#endif
	}

	cl_array1 = clCreateBuffer(
		context,
		CL_MEM_READ_WRITE,
		sizeof(T) * length,
		NULL,
		&status);

	if (status != CL_SUCCESS)
	{
		std::cout << "Error: clCreateBuffer (cl_array1)\n";
	}

	cl_array2 = clCreateBuffer(
		context,
		CL_MEM_READ_WRITE,
		sizeof(T) * length,
		NULL,
		&status);

	if (status != CL_SUCCESS)
	{
		std::cout << "Error: clCreateBuffer (cl_array2)\n";
	}
};


template <class T>
void MergeSortData<T>::Get_GPU_op(void)
{
	cl_int status, final_wg_size, num_wg, num_threads_needed_to_process;
	int enable_print = 0;
	size_t global_work_offset[3] = { 0 };
	size_t global_work_size[3] = { 0 };
	size_t local_work_size[3] = { 0 };

	cl_mem gpu_mem_inp;
	cl_mem gpu_mem_op;
	cl_mem gpu_mem_tmp;

	int merge_array_size;
	int array_size;
	int residual_array_size;
	int last_array_size;

	//copy cpu memory to GPU memory
	status = clEnqueueWriteBuffer(commands,
		cl_array1,
		BLOCKING_READ_WRITE,
		0,
		sizeof(T) * length,
		elems,
		0,
		NULL,
		NULL);
	if (status != CL_SUCCESS)
	{
		std::cout << "ERROR in cl_enqueue write";
	}

	//1 : Trigger the leaf level kernel

	//1.1 set input kernel arguments
	//input array
	status = clSetKernelArg(
		kernel_leaf,
		0,
		sizeof(cl_mem),
		(void *)&cl_array1);
	if (status != CL_SUCCESS)
	{
		std::cout << "Error: Setting kernel argument. (input)\n";
	}

	//final block size calculation
	final_wg_size = ((length % KERNEL_LEAF_WG_SZ) == 0)? KERNEL_LEAF_WG_SZ : (length % KERNEL_LEAF_WG_SZ);

	status = clSetKernelArg(
		kernel_leaf,
		1,
		sizeof(cl_int),
		(void *)&final_wg_size);
	if (status != CL_SUCCESS)
	{
		std::cout << "Error: Setting kernel argument. (input)\n";
	}


	local_work_size[0] = KERNEL_LEAF_WG_SZ;

	num_wg = length / KERNEL_LEAF_WG_SZ;
	if (final_wg_size != KERNEL_LEAF_WG_SZ) num_wg++;

	global_work_size[0] = num_wg * KERNEL_LEAF_WG_SZ;
	//1.2 fire the leaf level kernel
	status = clEnqueueNDRangeKernel(commands,
		kernel_leaf,
		1,
		global_work_offset,
		global_work_size,
		local_work_size,
		0,
		NULL,
		&kernel_event);

	clWaitForEvents(1, &kernel_event);

	if (status != CL_SUCCESS)
	{
		std::cout << "Error: kernel launch\n";
	}

	if (length <= KERNEL_LEAF_WG_SZ) return;

	/*stored in swapped order since swapping happens during start of while loop below*/
	gpu_mem_inp = cl_array2;
	gpu_mem_op = cl_array1;

	array_size = KERNEL_LEAF_WG_SZ;
	merge_array_size = (array_size << 1);

	num_threads_needed_to_process = length / merge_array_size;
	residual_array_size = length - (num_threads_needed_to_process * merge_array_size);

	last_array_size = array_size;
	if (residual_array_size > array_size)
	{
		last_array_size = residual_array_size - array_size;
		num_threads_needed_to_process += 1;
	}

	while (num_threads_needed_to_process > 0)
	{

		gpu_mem_tmp = gpu_mem_inp;
		gpu_mem_inp = gpu_mem_op;
		gpu_mem_op = gpu_mem_tmp;

		status = clSetKernelArg(
			kernel_merge,
			0,
			sizeof(cl_mem),
			(void *)&gpu_mem_inp);
		if (status != CL_SUCCESS)
		{
			std::cout << "Error: Setting kernel argument. (input)\n";
		}

		status = clSetKernelArg(
			kernel_merge,
			1,
			sizeof(cl_mem),
			(void *)&gpu_mem_op);
		if (status != CL_SUCCESS)
		{
			std::cout << "Error: Setting kernel argument. (output)\n";
		}

		status = clSetKernelArg(
			kernel_merge,
			2,
			sizeof(cl_int),
			(void *)&array_size);
		if (status != CL_SUCCESS)
		{
			std::cout << "Error: Setting kernel argument\n";
		}

		status = clSetKernelArg(
			kernel_merge,
			3,
			sizeof(cl_int),
			(void *)&last_array_size);
		if (status != CL_SUCCESS)
		{
			std::cout << "Error: Setting kernel argument\n";
		}

		status = clSetKernelArg(
			kernel_merge,
			4,
			sizeof(cl_int),
			(void *)&num_threads_needed_to_process);
		if (status != CL_SUCCESS)
		{
			std::cout << "Error: Setting kernel argument\n";
		}

		if (merge_array_size == 512)
		{
			enable_print = 1;
		}
		else
		{
			enable_print = 0;
		}

		

		status = clSetKernelArg(
			kernel_merge,
			5,
			sizeof(cl_int),
			(void *)&enable_print);
		if (status != CL_SUCCESS)
		{
			std::cout << "Error: Setting kernel argument\n";
		}

		local_work_size[0] = 16;
		global_work_size[0] = (num_threads_needed_to_process / 16) * 16;
		if (global_work_size[0] < num_threads_needed_to_process) global_work_size[0] += 16;


		status = clEnqueueNDRangeKernel(commands,
			kernel_merge,
			1,
			global_work_offset,
			global_work_size,
			local_work_size,
			0,
			NULL,
			&kernel_event);

		clWaitForEvents(1, &kernel_event);
		if (status != CL_SUCCESS)
		{
			std::cout << "Error: kernel launch\n";
		}

		array_size <<= 1;
		merge_array_size = (array_size << 1);

		num_threads_needed_to_process = length / merge_array_size;
		residual_array_size = length - (num_threads_needed_to_process * merge_array_size);

		last_array_size = array_size;
		if (residual_array_size > array_size)
		{
			last_array_size = residual_array_size - array_size;
			num_threads_needed_to_process += 1;
		}

	}

	cl_array1 = gpu_mem_op;
};


template <class T>
void MergeSortData<T>::Verify_GPU_op(void)
{
	int status, i;
	// read the output from GPU memory
	status = clEnqueueReadBuffer(commands,
		cl_array1,
		BLOCKING_READ_WRITE,
		0,
		sizeof(T) * length,
		(void *)elems_op_gpu,
		0,
		NULL,
		NULL);

	if (status != CL_SUCCESS)
	{
		std::cout << "Error: output copy\n";
	}

	// perform CPU merge
	MergeSort(elems,0,(length -1));


	for (i = 0; i < length; i++)
	{
		if (elems[i] != elems_op_gpu[i])
		{
			std::cout << "Mismatched at:" << i <<"\t"<< elems[i] << "\t"<<elems_op_gpu[i] << "\n";
		}
	}
#ifdef	ENABLE_DEBUG_PRINT
	std::cout << "The CPU output array is:\n";
	for (i = 0; i < length; i++)
	{
		std::cout << elems[i] << "\t";
	}

	std::cout << "The GPU output array is:\n";
	for (i = 0; i < length; i++)
	{
		std::cout << elems_op_gpu[i] << "\t";
	}

#endif	


};
template <class T>
void MergeSortData<T>::Merge(T tempArray[], int start, int mid, int end)
{
	T leftArray[5001];
	T rightArray[5001];
	int n1 = mid - start + 1;
	int n2 = end - mid;
	int i, j, k;

	for (i = 0; i < n1; i++)
	{
		leftArray[i] = tempArray[start + i];
	}
	for (j = 0; j < n2; j++)
	{
		rightArray[j] = tempArray[mid + j + 1];
	}
	// Assign the Sentinel value
	leftArray[i] = 99999;
	rightArray[j] = 99999;
	i = 0;
	j = 0;
	for (k = start; k <= end; k++)
	{
		if (leftArray[i] <= rightArray[j])
		{
			tempArray[k] = leftArray[i];
			i = i + 1;
		}
		else
		{
			tempArray[k] = rightArray[j];
			j = j + 1;
		}
	}
}

template <class T>
void MergeSortData<T>::MergeSort(T tempArray[], int start, int end)
{
	int mid;
	if (start < end)
	{
		mid = (start + end) / 2;
		MergeSort(tempArray, start, mid);
		MergeSort(tempArray, (mid + 1), end);
		Merge(tempArray, start, mid, end);
	}
}

/*
* Converts the contents of a file into a string
*/
std::string
convertToString(const char *filename)
{
	size_t size;
	char*  str;
	std::string s;

	std::fstream f(filename, (std::fstream::in | std::fstream::binary));

	if (f.is_open())
	{
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);

		str = new char[size + 1];
		if (!str)
		{
			f.close();
			std::cout << "Memory allocation failed";
			return NULL;
		}

		f.read(str, fileSize);
		f.close();
		str[size] = '\0';

		s = str;
		delete[] str;
		return s;
	}
	else
	{
		std::cout << "\nFile containg the kernel code(\".cl\") not found. Please copy the required file in the folder containg the executable.\n";
		exit(1);
	}
	return NULL;
}



int main(int argc, char** argv)
{
	int gpu = 1; //replace it with using argument initialization
	int err, is_double = 0;
	int input_size;

//	size_t global;                      // global domain size for our calculation
//	size_t local;                       // local domain size for our calculation

	cl_device_id device_id;             // compute device id 
	cl_program program;                 // compute program
	cl_uint numPlatforms;
	cl_platform_id platform = NULL;



	if (argc != 3)
	{
		//std::cout << "Usage is: executable.exe float/double input_array_size";
		//return EXIT_FAILURE;
		is_double = 1;
		input_size = 1023;
	}
	else
	{
		if (strcmp("double", "argv[1]") == 0)
		{
			is_double = 1;
		}

		input_size = atoi(argv[2]);
	}

	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (err != CL_SUCCESS)
	{
		std::cout << "Error: Getting Platforms. (clGetPlatformsIDs)\n";

	}

	if (numPlatforms > 0)
	{
		cl_platform_id* platforms = new cl_platform_id[numPlatforms];
		err = clGetPlatformIDs(numPlatforms, platforms, NULL);
		if (err != CL_SUCCESS)
		{
			std::cout << "Error: Getting Platform Ids. (clGetPlatformsIDs)\n";

		}
		for (unsigned int i = 0; i < numPlatforms; ++i)
		{
			char pbuff[100];
			err = clGetPlatformInfo(
				platforms[i],
				CL_PLATFORM_VENDOR,
				sizeof(pbuff),
				pbuff,
				NULL);
			if (err != CL_SUCCESS)
			{
				std::cout << "Error: Getting Platform Info.(clGetPlatformInfo)\n";
			}
			platform = platforms[i];
			if (!strcmp(pbuff, "Advanced Micro Devices, Inc."))
			{
				break;
			}
		}
		delete platforms;
	}

	if (NULL == platform)
	{
		std::cout << "NULL platform found so Exiting Application." << std::endl;
	}

	err = clGetDeviceIDs(platform, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	if (err != CL_SUCCESS)
	{
		std::cout << "Error: Failed to create a device group!\n";
		return EXIT_FAILURE;
	}

	// Create a compute context 
	//
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context)
	{
		std::cout << "Error: Failed to create a compute context!\n";
		return EXIT_FAILURE;
	}
	// Create a command commands
	//
	commands = clCreateCommandQueue(context, device_id, 0, &err);
	if (!commands)
	{
		std::cout << "Error: Failed to create a command commands!\n";
		return EXIT_FAILURE;
	}


	//Building Kernel
	const char * filename = "Template_Kernels.cl";
	std::string  sourceStr = convertToString(filename);
	const char * source = sourceStr.c_str();
	size_t sourceSize[] = { strlen(source) };

	program = clCreateProgramWithSource(
		context,
		1,
		&source,
		sourceSize,
		&err);
	if (err != CL_SUCCESS)
	{
		std::cout <<
			"Error: Loading Binary into cl_program \
               (clCreateProgramWithBinary)\n";
	}

	// Build the program executable
	//
	if (is_double)
	{
		err = clBuildProgram(program, 0, NULL, "-D DOUBLE", NULL, NULL);
	}
	else
	{
		err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	}
	if (err != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];

		std::cout << "Error: Failed to build program executable!\n";
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		std::cout << buffer << "\n" ;
		return EXIT_FAILURE;
	}

	// Create the compute kernel in the program we wish to run
	//
	kernel_leaf = clCreateKernel(program, "parallel_insertion_sort_leaf", &err);
	if (!kernel_leaf || err != CL_SUCCESS)
	{
		std::cout << "Error: Failed to create compute kernel_leaf!\n";
		return EXIT_FAILURE;
	}

	kernel_merge = clCreateKernel(program, "merge", &err);
	if (!kernel_merge || err != CL_SUCCESS)
	{
		std::cout << "Error: Failed to create compute kernel_leaf!\n";
		return EXIT_FAILURE;
	}

	if (is_double)
	{
		MergeSortData<double> merge_class_data(input_size);
		merge_class_data.Get_GPU_op();
		merge_class_data.Verify_GPU_op();
	}
	else
	{
		MergeSortData<float> merge_class_data(input_size);
		merge_class_data.Get_GPU_op();
		merge_class_data.Verify_GPU_op();
	}

}