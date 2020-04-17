//////////////////////////////////////////////////////////////
// Authors  : Lama Afra, Mohammed Al Khalidi, Taysseer Samman
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


// Swap two elements of an array
__device__ void swap_gpu(int* a, int* b)
{
	int temp = *a;
	*a = *b;
	*b = temp;
}

// A sequential version of the selection sort
// This algorithm will be applied after reaching the maximum recursion depth on gpu
__device__ void selectionSort(int* arr, int arrSize) 
{ 
    int i, j, min_idx; 
  
    // One by one move boundary of unsorted subarray 
    for (i = 0; i < arrSize - 1; i++) 
    { 
        // Find the minimum element in unsorted array 
        min_idx = i; 
        for (j = i+1; j < arrSize; j++)
        {
            if (arr[j] < arr[min_idx])
            {
                min_idx = j;
            }
        }
  
        // Swap the found minimum element with the first element 
        swap_gpu(&arr[min_idx], &arr[i]); 
    } 
}

// Computes the partition after rearranging the array
__device__ int partition_gpu(int* arr, int arrSize)
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
__global__ void quicksort_naive_kernel(int* arr, int arrSize, int depth)
{
    // If depth is more than maximum recursion
    // Apply sequential selection sort
    if(depth > MAX_RECURSION)
    {
        selectionSort(arr, arrSize);
        return;
    }

    // Partition
    int k = partition_gpu(arr, arrSize);

    if(k > 1) 
	{
        // Create cuda stream to run recursive calls in parallel
        cudaStream_t s_left;

        // Set the non-blocking flag for the cuda stream
        cudaStreamCreateWithFlags(&s_left, cudaStreamNonBlocking);

        // Sort the left partition
		quicksort_naive_kernel <<< 1, 1, 0, s_left >>> (&arr[0], k, depth + 1);

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
		quicksort_naive_kernel <<< 1, 1, 0, s_right >>> (&arr[k + 1], arrSize - k - 1, depth + 1);

        // Destroy the stream after getting done from it
        cudaStreamDestroy(s_right);
    }
}

// The partition kernel method
// The array size should be usually double the number of threads since each thread will be responsible for two array elements
__global__ void partition_kernel (
    int* arr,
    int* arrCopy,
    int* lessThan,
    int* greaterThan,
    int* lessThanSums,
    int* greaterThanSums,
    int* partitionArr,
    int* blockCounter,
    int* flags,
    int arrSize)
{
    // Shared memory
    __shared__ int bid_s;
    __shared__ int lsPrevSum_s;
    __shared__ int gtPrevSum_s;
    __shared__ int lsLocalSum_s;
    __shared__ int gtLocalSum_s;
    __shared__ int lessThan_s[2 * BLOCK_DIM];
    __shared__ int greaterThan_s[2 * BLOCK_DIM];

    // If this was the first thread
    if (threadIdx.x == 0)
    {
        //Get current block index and increment by 1
        bid_s = atomicAdd(&blockCounter[0], 1);
    }

    // Synchronize all threads
    __syncthreads();

    //Get the dynamic block id
    const int bid = bid_s;    

    // Load the real thread position
    int i = (2 * blockDim.x * bid) + threadIdx.x;

    // ========================= Copy to temporary, lessThan and greaterThan arrays =========================

    // Choose the middle element as the pivot
    int pivot = arr[(arrSize - 1) / 2];

    // Handle first element by the thread
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

    // Handle second element by the thread
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

    // ========================= Write partial sums =========================

    // Synchronize all threads
    __syncthreads();

    // If this was the last thread
    if (threadIdx.x == blockDim.x - 1)
    {
        lsLocalSum_s = lessThan_s[2 * BLOCK_DIM - 1];
        gtLocalSum_s = greaterThan_s[2 * BLOCK_DIM - 1];
    }

    // ========================= Single pass scan =========================

    // If this was the first thread
    if (threadIdx.x == 0)
    {
        // Wait for previous flag
        while (atomicAdd(&flags[bid], 0) == 0){;}
        
        // Check if there are blocks before
        if(bid > 0)
        {
            // Read previous partial sums
            lsPrevSum_s = lessThanSums[bid];
            gtPrevSum_s = greaterThanSums[bid];
        }
        else
        {
            // No previous sums, set to zero
            lsPrevSum_s = 0.0f;
            gtPrevSum_s = 0.0f;
        }

        // Propagate partial sum
        lessThanSums[bid + 1] = lsPrevSum_s + lsLocalSum_s;
        greaterThanSums[bid + 1] = gtPrevSum_s + gtLocalSum_s;

        // Memory fence
        __threadfence();

        // Set flag
        atomicAdd(&flags[bid + 1], 1);
    }

    // Synchronize all threads
    __syncthreads();

    // ========================= Commit changes to global memory =========================

    if (i < arrSize)
    {
        lessThan[i] = lessThan_s[threadIdx.x] + lsPrevSum_s + lsLocalSum_s;
        greaterThan[i] = greaterThan_s[threadIdx.x] + gtPrevSum_s + gtLocalSum_s;
    }

    if (i + blockDim.x < arrSize)
    {
        lessThan[i + blockDim.x] = lessThan_s[threadIdx.x + blockDim.x] + lsPrevSum_s + lsLocalSum_s;
        greaterThan[i + blockDim.x] = greaterThan_s[threadIdx.x + blockDim.x] + gtPrevSum_s + gtLocalSum_s;
    }
}

//Advanced version of the parallel quicksort which parallelizes both the partition method and the recursive calls
__global__ void quicksort_advanced_kernel(
    int* arr,
    int* arrCopy,
    int* lessThan,
    int* greaterThan,
    int* lessThanSums,
    int* greaterThanSums,
    int* partitionArr,
    int* blockCounter,
    int* flags,
    int depth,
    int arrSize)
{
    // If depth is more than maximum recursion
    // Apply sequential selection sort
    if(depth > MAX_RECURSION)
    {
        selectionSort(arr, arrSize);
        return;
    }

    // Configure the number of blocks and threads per block
    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = 2 * numThreadsPerBlock;
    const unsigned int numBlocks = (arrSize + numElementsPerBlock - 1)/numElementsPerBlock;

	// Partition
    partition_kernel <<< numBlocks, numThreadsPerBlock >>> (arr, arrCopy, lessThan, greaterThan, lessThanSums, greaterThanSums, partitionArr, blockCounter, flags, arrSize);

    // Set partition as first element of the array after the partition kernel has done its work
    int k = partitionArr[0];

    if(k > 1)
	{
        // Create cuda stream to run recursive calls in parallel
        cudaStream_t s_left;

        // Set the non-blocking flag for the cuda stream
        cudaStreamCreateWithFlags(&s_left, cudaStreamNonBlocking);

        // Sort the left partition
		quicksort_advanced_kernel <<< 1, 1, 0, s_left >>> (
            &arr[0],
            &arrCopy[0],
            &lessThan[0],
            &greaterThan[0],
            &lessThanSums[0],
            &greaterThanSums[0],
            &partitionArr[0],
            &blockCounter[0],
            &flags[0],
            depth + 1,
            k
        );

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
		quicksort_advanced_kernel <<< 1, 1, 0, s_right >>> (
            &arr[k + 1],
            &arrCopy[k + 1],
            &lessThan[k + 1],
            &greaterThan[k + 1],
            &lessThanSums[k + 1],
            &greaterThanSums[k + 1],
            &partitionArr[k + 1],
            &blockCounter[k + 1],
            &flags[k + 1],
            depth + 1,
            arrSize - k - 1
        );

        // Destroy the stream after getting done from it
        cudaStreamDestroy(s_right);
    }
}

__host__ void quicksort_gpu(int* arr, int arrSize)
{
    //Define the timer
    Timer timer;

    //Allocate GPU memory
    startTime(&timer);
    
    //Declare and allocate required arrays on the device
    int* arr_d;
    int* arrCopy;
    int* lessThan;
    int* greaterThan;
    int* lessThanSums;
    int* greaterThanSums;
    int* partitionArr;
    int* blockCounter;
    int* flags;

    cudaMalloc((void**) &arr_d, arrSize * sizeof(int));
    cudaMalloc((void**) &arrCopy, arrSize * sizeof(int));
    cudaMalloc((void**) &lessThan, arrSize * sizeof(int));
    cudaMalloc((void**) &greaterThan, arrSize * sizeof(int));
    cudaMalloc((void**) &lessThanSums, arrSize * sizeof(int));
    cudaMalloc((void**) &greaterThanSums, arrSize * sizeof(int));
    cudaMalloc((void**) &partitionArr, arrSize * sizeof(int));
    cudaMalloc((void**) &blockCounter, arrSize * sizeof(int));
    cudaMalloc((void**) &flags, arrSize * sizeof(int));

    ////Initialize all block counters to 0
    //for (unsigned int i = 0; i < arrSize; i++)
    //{
    //    blockCounter[i] = 0;
    //}

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    //Copy data to GPU
    startTime(&timer);
    
    //Copy data for the array from host to device
    cudaMemcpy(arr_d, arr, arrSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    //Call kernel
    startTime(&timer);

    //Sorting on GPU
    if(arrSize > 1) 
	{
        //quicksort_advanced_kernel << < 1, 1, 0 >> > (arr_d, arrCopy, lessThan, greaterThan, partitionArr, lessThanSums, greaterThanSums, blockCounter, flags, 1, arrSize);
        quicksort_naive_kernel << < 1, 1, 0 >> > (arr_d, arrSize, 1);
    }

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time");

    //Copy data from GPU
    startTime(&timer);
    
    //After performing the quick sort, copy the sorted array from device to host
    cudaMemcpy(arr, arr_d, arrSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    //Free GPU memory
    startTime(&timer);
    
    //Now that we are done, we can free the allocated memory to leave space for other computations
    cudaFree(arr_d);
    cudaFree(arrCopy);    
    cudaFree(lessThan);
    cudaFree(greaterThan);
    cudaFree(lessThanSums);
    cudaFree(greaterThanSums);
    cudaFree(partitionArr);
    cudaFree(blockCounter);
    cudaFree(flags);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");
}