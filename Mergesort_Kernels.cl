/*
This is a parallel merge sort code
*/

#define WGSIZE 256

#ifdef DOUBLE
#define FPTYPE double

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "Double precision floating point not supported by OpenCL implementation."
#endif
#else
#define FPTYPE float
#endif

kernel void parallel_insertion_sort_leaf(
	__global FPTYPE* g_array,
	int final_blk_size
	)
{
	int g_id = get_global_id(0);
	int grp_id = get_group_id(0);
	int grp_size = get_num_groups(0);
	int l_id = get_local_id(0);
	int l_size = get_local_size(0);
	__local FPTYPE l_array[WGSIZE];
	int iter;
	int final_idx, repeat_cnt;
	int local_blk_size = WGSIZE;
	FPTYPE my_element;

	if (grp_id == (grp_size - 1))
	{
		local_blk_size = final_blk_size;
	}

	if (l_id < local_blk_size)
	{
		my_element = g_array[g_id];
		l_array[l_id] = my_element;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (l_id >= local_blk_size) return;

	final_idx = 0;
	repeat_cnt = -1;
	for (iter = 0; iter < local_blk_size; iter++)
	{
		final_idx += (l_array[iter] < my_element)? 1 : 0;
		repeat_cnt += (l_array[iter] == my_element) ? 1 : 0;
	}

	//store the results

	final_idx += grp_id * l_size;
	g_array[final_idx] = my_element;

//	printf("[%d] %d %d %f\n",get_global_id(0), final_idx, repeat_cnt, my_element);
	for (iter = 0; iter < repeat_cnt; iter++)
	{
		g_array[++final_idx] = my_element;
	}
}

kernel void merge(
	__global FPTYPE* g_array_inp,
	__global FPTYPE* g_array_op,
	int array_size,
	int last_array_size,
	int tot_thread_needed,
	int enable_print
	)
{
	int g_id = get_global_id(0);
	int my_start_idx_1 = g_id * (array_size * 2);
	int my_start_idx_2 = my_start_idx_1 + array_size;
	int op_idx = my_start_idx_1;
	int array_1_len = array_size;
	int array_2_len = array_size;
	int iter;
	int my_idx_1, my_idx_2;

	FPTYPE array1_element, array2_element;

	if (g_id >= tot_thread_needed) return;
	if (g_id == (tot_thread_needed - 1))
	{
		array_2_len = last_array_size;
	}

	my_idx_1 = my_start_idx_1;
	my_idx_2 = my_start_idx_2;

	array1_element = g_array_inp[my_idx_1++];
	array2_element = g_array_inp[my_idx_2++];

	/*if ((g_id == 0) && (enable_print == 1))
	{
		for (iter = 0; iter < (1024); iter++)
		{
			if ((iter % 256) == 0)
			{
				printf("\n***************[%d]*******************\n",g_id);
			}
			printf("%f\t", g_array_inp[iter]);

		}
	}*/
	//printf("[%d] %d %d %f %f %d [%d %d]\n",g_id, my_start_idx_1, my_start_idx_2, array1_element, array2_element, op_idx, array_1_len, array_2_len);
	for (iter = 0; iter < (array_1_len + array_2_len); iter++)
	{
		if (array1_element < array2_element)
		{
			g_array_op[op_idx++] = array1_element;

			if (my_idx_1 < my_start_idx_2)
			{
				array1_element = g_array_inp[my_idx_1++];
			}
			else
			{
				iter++;
				g_array_op[op_idx++] = array2_element;
				iter++;
				for (; iter < (array_1_len + array_2_len); iter++)
				{
					g_array_op[op_idx++] = g_array_inp[my_idx_2++];
				}
				break;
			}
		}
		else
		{
			g_array_op[op_idx++] = array2_element;
			if (my_idx_2 < (my_start_idx_2 + array_2_len))
			{
				array2_element = g_array_inp[my_idx_2++];
			}
			else
			{
				iter++;
				g_array_op[op_idx++] = array1_element;
				iter++;
				for (; iter < (array_1_len + array_2_len); iter++)
				{
					g_array_op[op_idx++] = g_array_inp[my_idx_1++];
				}
				break;
			}
		}

	}
	//printf("[%d]op_idx = %d %f\n",g_id, op_idx, g_array_op[512]);
}
