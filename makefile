NVCC = nvcc
CFLAGS = -g -G -O0
mm_cuda: kernel.cu main.h
	$(NVCC) $(CFLAGS) $< -o $@