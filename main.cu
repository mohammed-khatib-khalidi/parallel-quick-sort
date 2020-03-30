#include "common.h"
#include "timer.h"

//Swap two elements of an array
void swap_cpu(float* a, float* b)
{
	float temp = *a;
	*a = *b;
	*b = temp;
}

//Computes the partition after rearranging the array
int partition_cpu(float* arr, int start, int end)
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
			swap_cpu(&arr[i], &arr[j]);
		}
	}

	//Swap array elements with indices i + 1 and pivot
    swap_cpu(&arr[i + 1], &arr[end]);

	//Return parition index
    return (i + 1);
}

//Sorts an array with the quick sort algorithm
void quicksort_cpu(float* arr, int start, int end)
{
	//Array size must be positive
	if (start < end)
	{
		//Partition
        int k = partition_cpu(arr, start, end);

		//Sort the left partition
        quicksort_cpu(arr, start, k - 1);

		//Sort the right partition
        quicksort_cpu(arr, k + 1, end);
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
    for (unsigned int i = 0; i < arrSize; ++i) {
        float val = rand();
        arr_cpu[i] = val;
        arr_gpu[i] = val;
    }

    // Compute on CPU
    startTime(&timer);
    quicksort_cpu(arr_cpu, 0, arrSize - 1);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time");

    // Compute on GPU
    startTime(&timer);
    quicksort_gpu(arr_gpu, arrSize);
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
    for(unsigned int i = 0; i < arrSize; ++i) {
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

