#include <stdio.h>
#define ARRAY_SIZE 7

//Swap two elements of an array
void swap(int* a, int* b)
{
	int temp = *a;
	*a = *b;
	*b = temp;
}

//Computes the partition after rearranging the array
int partition_cpu(int* arr, int start, int end, int pivot)
{
	//Index of smaller element
	int i = start - 1;

	for (int j = start; j < end; j++)
	{
		//If current element is smaller than the pivot
		if (arr[j] < arr[pivot])
		{
			//Increment the index of the smaller element
			i++;
			//Swap array elements with indices i and j
			swap(&arr[i], &arr[j]);
		}
	}

	//Swap array elements with indices i + 1 and pivot
	swap(&arr[i + 1], &arr[pivot]);

	//Return parition index
	return (i + 1);
}

//Sorts an array with the quick sort algorithm
void quicksort_cpu(int* arr, int start, int end)
{
	//Array size must be positive
	if (start < end)
	{
		//Choose pivot to be the last element of the array
		int pivot = end;

		//Partition
		int partition = partition_cpu(arr, start, end, pivot);

		//Sort the left partition
		quicksort_cpu(arr, start, partition - 1);

		//Sort the right partition
		quicksort_cpu(arr, partition + 1, end);
	}

}

int main(int argc, char** argv)
{
	int inputArray[ARRAY_SIZE] = { 10, 80, 30, 90, 40, 50, 70 };

	////Compute array size
	//int arraySize = sizeof(inputArray) / sizeof(inputArray[0]);

	//Sort the array
	quicksort_cpu(inputArray, 0, ARRAY_SIZE - 1);

	//Print the sorted array
	printf("\nPrint array elements after sorting:\n");
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		printf("%d ", inputArray[i]);
	}

	return;
}