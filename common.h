#define BLOCK_DIM 1024
#define ARRAY_SIZE 10240
#define ARRAY_THRESHOLD 5120
#define MAX_RECURSION 24

void quicksort_gpu(int* array, int arraySize);
void partition_gpu(int* arr, int* partitionIdx, int arrSize);