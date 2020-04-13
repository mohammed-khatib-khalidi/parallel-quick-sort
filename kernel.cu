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

#define BLOCK_DIM 1024

// The partition kernel method
// The array size should be usually double the number of threads since each thread will be responsible for two array elements
__global__ void partition_kernel(float* arr, float* arrCopy, float* lessThan, float* greaterThan, int* partArr, int arrSize)
{
    // Shared memory buffer
    __shared__ float lessThan_s[2 * BLOCK_DIM];
    __shared__ float greaterThan_s[2 * BLOCK_DIM];

    // Load the real thread position
    int i = (2 * blockDim.x * blockIdx.x) + threadIdx.x;

	// In case the array was only one item then return the start index
	if (arrSize == 1) 
	{
        // Allow only the first thread to modify partition k
        if (threadIdx.x == 0)
        {
            partArr[i] = 0;
        }
        // Stop here
        return;
    }

    // ========================= Copy to temporary, lessThan and greaterThan arrays =========================

    // Choose the middle element as the pivot
    float pivot = arr[(arrSize - 1) / 2];

    // First element the thread is responsible for
    if(i < arrSize)
    {
        // Copy to temporary array
        arrCopy[i] = arr[i];

        // Copy to the lessThan array
        if(arr[i] < pivot)
        {
            lessThan_s[i] = 1;
        }
        else
        {
            lessThan_s[i] = 0;
        }

        // Copy to the greaterThan array
        if(arr[i] > pivot)
        {
            greaterThan_s[i] = 1;
        }
        else
        {
            greaterThan_s[i] = 0;
        }
    }

    // Second element the thread is responsible for
    if(i + blockDim.x < arrSize)
    {
        arrCopy[i + blockDim.x] = arr[i + blockDim.x];

        // Copy to the lessThan array
        if(arr[i + blockDim.x] < pivot)
        {
            lessThan_s[i + blockDim.x] = 1;
        }
        else
        {
            lessThan_s[i + blockDim.x] = 0;
        }

        // Copy to the greaterThan array
        if(arr[i + blockDim.x] > pivot)
        {
            greaterThan_s[i + blockDim.x] = 1;
        }
        else
        {
            greaterThan_s[i + blockDim.x] = 0;
        }
    }

    // ========================= Prefix sum (lessThan & greaterThan) =========================

    // ========================= Reduction phase =========================

    for(unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {       
        // Synchronize all threads
        __syncthreads();
        // Re-index threads to minimize divergence
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if(index >= stride && index < 2 * blockDim.x) {
            lessThan_s[index] += lessThan_s[index - stride];
            greaterThan_s[index] += greaterThan_s[index - stride];
        }
    }

    // ========================= Post-Reduction phase =========================

    for (int stride = BLOCK_DIM / 2; stride > 0; stride /= 2)
    {
        // Synchronize all threads
        __syncthreads();
        // Re-index threads to minimize divergence
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if(index + stride < 2 * blockDim.x)
        {
            lessThan_s[index + stride] += lessThan_s[index];
            greaterThan_s[index + stride] += greaterThan_s[index];
        }
    }

    // Synchronize all threads
    __syncthreads();
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

	// Return partition index
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
__global__ void quicksort_advanced_kernel(float* arr, float* arrCopy, float* lessThan, float* greaterThan, int* partArr, int arrSize)
{  
    // Configure the number of blocks and threads per block
    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = 2 * numThreadsPerBlock;
    const unsigned int numBlocks = (arrSize + numElementsPerBlock - 1)/numElementsPerBlock;

	// Partition
    partition_kernel <<< numBlocks, numThreadsPerBlock >>> (arr, arrCopy, lessThan, greaterThan, partArr, arrSize);

    // Set partition as first element of the array after the partition kernel has done its work
    int k = partArr[0];

    if(k > 1) 
	{
        // Create cuda stream to run recursive calls in parallel
        cudaStream_t s_left;

        // Set the non-blocking flag for the cuda stream
        cudaStreamCreateWithFlags(&s_left, cudaStreamNonBlocking);

        // Sort the left partition
		quicksort_advanced_kernel <<< 1, 1, 0, s_left >>> (&arr[0], &arrCopy[0], &lessThan[0], &greaterThan[0], &partArr[0], k);

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
		quicksort_advanced_kernel <<< 1, 1, 0, s_right >>> (&arr[k + 1], &arrCopy[k + 1], &lessThan[k + 1], &greaterThan[k + 1], &partArr[k + 1], arrSize - k - 1);

        // Destroy the stream after getting done from it
        cudaStreamDestroy(s_right);
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
    int* partArr_d;
    cudaMalloc((void**) &arr_d, arrSize * sizeof(float));
    cudaMalloc((void**) &arrCopy_d, arrSize * sizeof(float));
    cudaMalloc((void**) &lessThan_d, arrSize * sizeof(float));
    cudaMalloc((void**) &greaterThan_d, arrSize * sizeof(float));
    cudaMalloc((void**) &partArr_d, arrSize * sizeof(int));

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
        //quicksort_advanced_kernel << < 1, 1, 0 >> > (arr_d, arrCopy_d, lessThan_d, greaterThan_d, partArr_d, arrSize);
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