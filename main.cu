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

// This method will take a string and trim any extra whitespace
char* trimWhiteSpace(char *str)
{
	char *end;

	// Trim leading space
	while (((char)*str == ' ' || (char)*str == '\t' || (char)*str == '\n' || (char)*str == '\r')) str++;

	if (*str == 0)  // All spaces?
		return str;

	// Trim trailing space
	end = str + strlen(str) - 1;
	while (end > str && ((char)*end == ' ' || (char)*end == '\t' || (char)*end == '\n' || (char)*end == '\r')) end--;

	// Write new null terminator character
	end[1] = '\0';

	return str;
}

// Swap two elements of an array
void swap_cpu(int* a, int* b)
{
	int temp = *a;
	*a = *b;
	*b = temp;
}

// Computes the partition after rearranging the array
int partition_cpu(int* arr, int arrSize)
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
void quicksort_cpu(int* arr, int arrSize)
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

// Main method
int main(int argc, char**argv)
{
    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
	unsigned int arrSize = (argc > 1) ? (atoi(argv[1])) : ARRAY_SIZE;
	int* arr_cpu = (int*) malloc(arrSize * sizeof(int));
    int* arr_gpu = (int*) malloc(arrSize * sizeof(int));
	int* arr_init = (int*)malloc(arrSize * sizeof(int));

	// Input arguments to help with debugging and program launching
	// Index 0: debug-on | debug-off
	// Index 1: naive	 | advanced
	int inputArgumentCount = 0;
	char** inputArguments = (char**) malloc(argc * (10 * sizeof(char)));
	if (argc > 2)
	{
		int i = 2;
		while (argc > i)
		{
			inputArguments[i - 2] = (char*)malloc(strlen(argv[i]) * sizeof(char));
			inputArguments[i - 2] = argv[i];
			
			i++;
			inputArgumentCount++;
		}
	}
	
	// Initialize array with a list of random numbers
	for (unsigned int i = 0; i < arrSize; ++i) 
	{
        int val = rand() % 1000 + 1;
        arr_cpu[i] = val;
		arr_gpu[i] = val;
		arr_init[i] = val;
    }

	printf("\n");

	// Compute on CPU
    startTime(&timer);
	quicksort_cpu(arr_cpu, arrSize);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time");

    // Compute on GPU
    startTime(&timer);
	quicksort_gpu(arr_gpu, arrSize, inputArgumentCount, inputArguments);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time");

	// Debugging (condition: debug_on = true)
	if (inputArgumentCount > 0 && strcmp(inputArguments[0], "debug-on") == 0)
	{
		printf("\nStarting arguments:\n");
		for (int i = 0; i < inputArgumentCount; i++)
		{
			printf("%d: %s \n", i, inputArguments[i]);
		}

		printf("\n");
		printf("Initial Array:\t");
		for (unsigned int i = 0; i < arrSize; ++i)
		{
			printf("%d ", arr_init[i]);
		}

		printf("\n");
		printf("CPU Array:\t");
		for (unsigned int i = 0; i < arrSize; ++i) 
		{
			printf("%d ", arr_cpu[i]);
		}

		printf("\n");
		printf("GPU Array:\t");
		for (unsigned int i = 0; i < arrSize; ++i) 
		{
			printf("%d ", arr_gpu[i]);
		}
		printf("\n\n");
	}

    // Verify result
    for(unsigned int i = 0; i < arrSize; ++i) 
	{
        if(arr_cpu[i] != arr_gpu[i])
        {
            printf("Mismatch at index %u (CPU result = %d, GPU result = %d)\n", i, arr_cpu[i], arr_gpu[i]);
            exit(0);
        }
    }

    // Free memory
    free(arr_cpu);
	free(arr_gpu);
	free(arr_init);

    //Exit program
    return 0;
}

