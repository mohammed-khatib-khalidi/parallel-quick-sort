
NVCC    = nvcc -arch=sm_35 -rdc=true
OBJ     = main.o kernel.o
DEPS    = quicksort.h timer.h
EXE     = quicksort


default: $(EXE)

%.o: %.cu
	$(NVCC) -c -o $@ $<

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)

