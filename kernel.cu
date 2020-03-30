#include "common.h"
#include "timer.h"

//Swap two elements of an array
__device__  void swap_gpu(float* a, float* b)
{
	float temp = *a;
	*a = *b;
	*b = temp;
}

//Computes the partition after rearranging the array
__device__ int partition_gpu(float* arr, int start, int end)
{
	//Index of smaller element
	int i = start - 1;

	for (int j = start; j < end; j++)
	{
		//If current element is smaller than the pivot
		if (arr[j] < arr[end])
		{
			//Increment the index of the smaller element
			i++;
			//Swap array elements with indices i and j
			swap_gpu(&arr[i], &arr[j]);
		}
	}

	//Swap array elements with indices i + 1 and pivot
	swap_gpu(&arr[i + 1], &arr[end]);

	//Return parition index
	return (i + 1);
}

__global__ void quicksort_naive_kernel(float* arr, int start, int end)
{
    //Partition
    int k = partition_gpu(arr, start, end);

    //Sort the left partition
    if(start < k - 1) {
        quicksort_naive_kernel <<< 1, 1 >>> (arr, start, k - 1);
    }

    //Sort the right partition
    if(k + 1 < end) {
        quicksort_naive_kernel <<< 1, 1 >>> (arr, k + 1, end);
    }
}

void quicksort_gpu(float* arr, int arrSize)
{
    //Define the timer
    Timer timer;

    //Allocate GPU memory
    startTime(&timer);
    
    //Declare and allocate the same array on the device
    float *arr_d;
    cudaMalloc((void**) &arr_d, arrSize * sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    //Copy data to GPU
    startTime(&timer);
    
    //Copy data for the array from host to device
    cudaMemcpy(arr_d, arr, arrSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    //Call kernel
    startTime(&timer);
        
    // Configure the number of blocks and threads per block
    //const unsigned int numThreadsPerBlock = 512;
    //const unsigned int numBlocks = (arrSize + numThreadsPerBlock - 1)/numThreadsPerBlock;

    //Sorting on GPU
    if(arrSize > 1) {
        quicksort_naive_kernel <<< 1, 1 >>> (arr_d, 0, arrSize - 1);
    }

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time");

    //Copy data from GPU
    startTime(&timer);
    
    //After performing the quick sort, copy the sorted array from device to host
    cudaMemcpy(arr, arr_d, arrSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    //Free GPU memory
    startTime(&timer);
    
    //Now that we are done, we can free the allocated memory to leave space for other computations
    cudaFree(arr_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");
}