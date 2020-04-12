//////////////////////////////////////////////////////////////
// Authors  : Lama Afra, Mohammad Al Khalidi, Taysseer Samman
// Usernames: laa59, mwa30, tjs00
// Course   : CMPS 396AA
// Timestamp: 20200328
// Project  : Parallel Quicksort
/////////////////////////////////////////////////////////////

#include "common.h"
#include "timer.h"

// CUDA Windows Headers
#if defined _WIN32 || defined _WIN64
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

// The partition kernel method
__global__ void partition_kernel(float* arr, float* arrCopy, float* lessThan, float* greaterThan, int start, int end, int pivotIdx, int k)
{
	// Calculate the size of the array
    int arrSize = end - start + 1;
    
    // Compute the thread index
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Compute the real array index
    int index = start + tid;

	// In case the array was only one item then return the start index
	if (arrSize == 1) 
	{
        // Allow only the first thread to modify partition k
        if (tid == 0) 
			k = start;
        // Stop here
        return;
    }

    // Compute the pivot value
    float pivot = arr[pivotIdx];

    // Copy to temporary array
    arrCopy[tid] = arr[index];

    // Copy to the lessThan array
    if(arr[index] < pivot)
        lessThan[tid] = 1;
    else
        lessThan[tid] = 0;

    // Copy to the greaterThan array
    if(arr[index] > pivot) 
        greaterThan[tid] = 1;
    else
        greaterThan[tid] = 0;

    // Sync all threads
    __syncthreads();

    // Now we will start performing the prefix sum for the lessThan and greaterThan arrays
}

// Swap two elements of an array
__device__ void swap_gpu(float* a, float* b)
{
	float temp = *a;
	*a = *b;
	*b = temp;
}

// Computes the partition after rearranging the array
__device__ int partition_gpu(float* arr, int arrSize)
{
	// Index of smaller element
    int i = - 1;

	for (int j = 0; j < arrSize - 1; j++)
	{
		// If current element is smaller than the pivot
		if (arr[j] < arr[arrSize - 1])
		{
			// Increment the index of the smaller element
			i++;
			// Swap array elements with indices i and j
			swap_gpu(&arr[i], &arr[j]);
		}
	}

	// Swap array elements with indices i + 1 and pivot
	swap_gpu(&arr[i + 1], &arr[arrSize - 1]);

	// Return parition index
    return (i + 1);
}

// Naive version of the parallel quicksort which only parallelizes recursive calls
__global__ void quicksort_naive_kernel(float* arr, int arrSize)
{
    // Partition
    int k = partition_gpu(arr, arrSize);

    if(k > 1) 
	{
        // Create cuda stream to run recursive calls in parallel
        cudaStream_t s_left;

        // Set the non-blocking flag for the cuda stream
        cudaStreamCreateWithFlags(&s_left, cudaStreamNonBlocking);

        // Sort the left partition
		quicksort_naive_kernel <<< 1, 1, 0, s_left >>> (&arr[0], k);

        // Destroy the stream after getting done from it
        cudaStreamDestroy(s_left);
    }

    if(arrSize > k + 2) 
	{
        // Create cuda stream to run recursive calls in parallel
        cudaStream_t s_right;

        // Set the non-blocking flag for the cuda stream
        cudaStreamCreateWithFlags(&s_right, cudaStreamNonBlocking);

        // Sort the right partition
		quicksort_naive_kernel <<< 1, 1, 0, s_right >>> (&arr[k + 1], arrSize - k - 1);

        // Destroy the stream after getting done from it
        cudaStreamDestroy(s_right);
    }
}

//Advanced version of the parallel quicksort which parallelizes both the partition method and the recursive calls
__global__ void quicksort_advanced_kernel(float* arr, int start, int end)
{
	// Get size of the array
	int arrSize = end - start + 1;

	// Allocate memory for the three arrays
	float* arrCopy;
	float* lessThan;
	float* greaterThan;

	cudaMalloc((void**)&arrCopy, arrSize * sizeof(float));
	cudaMalloc((void**)&lessThan, arrSize * sizeof(float));
	cudaMalloc((void**)&lessThan, arrSize * sizeof(float));

	int pivotIdx = (start + end) / 2;

	// Configure the number of blocks and threads per block
	const unsigned int numThreadsPerBlock = 512;
	const unsigned int numBlocks = (arrSize + numThreadsPerBlock - 1) / numThreadsPerBlock;

	// Partition
	int k = 0;
	partition_kernel << < numBlocks, numThreadsPerBlock >> > (arr, arrCopy, lessThan, greaterThan, start, end, pivotIdx, k);

	// Sort the left partition
	if (start < k - 1) 
	{
		quicksort_advanced_kernel << < 1, 1 >> > (arr, start, k - 1);
	}

	// Sort the right partition
	if (k + 1 < end) 
	{
		quicksort_advanced_kernel << < 1, 1 >> > (arr, k + 1, end);
	}
}

__host__ void quicksort_gpu(float* arr, int arrSize)
{
    //Define the timer
    Timer timer;

    //Allocate GPU memory
    startTime(&timer);
    
    //Declare and allocate required arrays on the device
    float* arr_d;
    float* arrCopy_d;
    float* lessThan_d;
    float* greaterThan_d;
    float* partition_d;
    cudaMalloc((void**) &arr_d, arrSize * sizeof(float));
    cudaMalloc((void**) &arrCopy_d, arrSize * sizeof(float));
    cudaMalloc((void**) &lessThan_d, arrSize * sizeof(float));
    cudaMalloc((void**) &greaterThan_d, arrSize * sizeof(float));
    cudaMalloc((void**) &partition_d, arrSize * sizeof(float));

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

    //Sorting on GPU
    if(arrSize > 1) 
	{
		quicksort_naive_kernel << < 1, 1, 0 >> > (arr_d, arrSize);
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
    cudaFree(arrCopy_d);
    cudaFree(lessThan_d);
    cudaFree(greaterThan_d);
    cudaFree(partition_d);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");
}