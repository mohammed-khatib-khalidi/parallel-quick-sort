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
void swap_cpu(float* a, float* b)
{
	float temp = *a;
	*a = *b;
	*b = temp;
}

// Computes the partition after rearranging the array
int partition_cpu(float* arr, int arrSize)
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
			swap_cpu(&arr[i], &arr[j]);
		}
	}

	// Swap array elements with indices i + 1 and pivot
	swap_cpu(&arr[i + 1], &arr[arrSize - 1]);

	// Return parition index
    return (i + 1);
}

// Sorts an array with the quick sort algorithm
void quicksort_cpu(float* arr, int arrSize)
{
	// Array size must be greater than 1
	if (arrSize > 1)
	{
		// Partition
        int k = partition_cpu(arr, arrSize);

		// Sort the left partition
		quicksort_cpu(&arr[0], k);

		// Sort the right partition
		quicksort_cpu(&arr[k + 1], arrSize - k - 1);
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

    //Global array which will be used by the partition kernel
    float* arrCopy_gpu = (float*) malloc(arrSize * sizeof(float));
    float* lessThan_gpu = (float*) malloc(arrSize * sizeof(float));
    float* greaterThan_gpu = (float*) malloc(arrSize * sizeof(float));
    float* partition_gpu = (float*) malloc(arrSize * sizeof(float));
    
	for (unsigned int i = 0; i < arrSize; ++i) 
	{
        float val = rand();
        arr_cpu[i] = val;
        arr_gpu[i] = val;
    }

    // Compute on CPU
    startTime(&timer);
	quicksort_cpu(arr_cpu, arrSize);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time");

    // Compute on GPU
    startTime(&timer);
	quicksort_gpu(arr_gpu, arrSize); //arrCopy_gpu, lessThan_gpu, greaterThan_gpu
    stopTime(&timer);
    printElapsedTime(timer, "GPU time");

    printf("\n");
    for (unsigned int i = 0; i < arrSize; ++i) {
        printf("%e ", arr_cpu[i]);
    }
    printf("\n");

    printf("\n");
    for (unsigned int i = 0; i < arrSize; ++i) {
        printf("%e ", arr_gpu[i]);
    }
    printf("\n");

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

    //Exit program
    return 0;
}

