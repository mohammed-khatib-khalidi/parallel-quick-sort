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

// Swap two elements of an array
void Swap_CPU(float* a, float* b)
{
	float temp = *a;
	*a = *b;
	*b = temp;
}

// Computes the partition after rearranging the array
int Partition_CPU(float* arr, int start, int end)
{
	// Index of smaller element
    int i = start - 1;

	for (int j = start; j < end; j++)
	{
		// If current element is smaller than the pivot
		if (arr[j] < arr[end])
		{
			// Increment the index of the smaller element
			i++;
			// Swap array elements with indices i and j
			Swap_CPU(&arr[i], &arr[j]);
		}
	}

	// Swap array elements with indices i + 1 and pivot
	Swap_CPU(&arr[i + 1], &arr[end]);

	// Return parition index
    return (i + 1);
}

// Sorts an array with the quick sort algorithm
void Quicksort_CPU(float* arr, int start, int end)
{
	// Array size must be positive
	if (start < end)
	{
		// Partition
        int k = Partition_CPU(arr, start, end);

		// Sort the left partition
		Quicksort_CPU(arr, start, k - 1);

		// Sort the right partition
		Quicksort_CPU(arr, k + 1, end);
	}
}

int main(int argc, char**argv)
{
    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    unsigned int arrSize = (argc > 1)?(atoi(argv[1])):ARRAY_SIZE;
    float* arr_cpu = (float*) malloc(arrSize * sizeof(float));
    float* arr_gpu = (float*) malloc(arrSize * sizeof(float));
    
	for (unsigned int i = 0; i < arrSize; ++i) 
	{
        float val = rand();
        arr_cpu[i] = val;
        arr_gpu[i] = val;
    }

    // Compute on CPU
    startTime(&timer);
	Quicksort_CPU(arr_cpu, 0, arrSize - 1);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time");

    // Compute on GPU
    startTime(&timer);
	Quicksort_GPU(arr_gpu, arrSize);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time");

    //printf("\n");
    //for (unsigned int i = 0; i < arrSize; ++i) {
    //    printf("%e ", arr_cpu[i]);
    //}
    //printf("\n");

    //printf("\n");
    //for (unsigned int i = 0; i < arrSize; ++i) {
    //    printf("%e ", arr_gpu[i]);
    //}
    //printf("\n");

    // Verify result
    for(unsigned int i = 0; i < arrSize; ++i) 
	{
        if(arr_cpu[i] != arr_gpu[i])
        {
            printf("Mismatch at index %u (CPU result = %e, GPU result = %e)\n", i, arr_cpu[i], arr_gpu[i]);
            exit(0);
        }
    }

    // Free memory
    free(arr_cpu);
    free(arr_gpu);

    return 0;
}

