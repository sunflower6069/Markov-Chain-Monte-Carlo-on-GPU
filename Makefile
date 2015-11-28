CUDAPATH = /home/apps/fas/GPU/cuda_6.0.37

NVCC = $(CUDAPATH)/bin/nvcc

NVCCFLAGS = -I$(CUDAPATH)/include -O3

LFLAGS = -L$(CUDAPATH)/lib64 -lcuda -lcudart -lm

# Compiler-specific flags (by default, we always use sm_20)
GENCODE_SM20 = -gencode=arch=compute_20,code=\"sm_20,compute_20\"
GENCODE = $(GENCODE_SM20)

.SUFFIXES : .cu .ptx

BINARIES = MCMCpar

MCMCpar: MCMCpar.o 
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $^

.cu.o:
	$(NVCC) $(GENCODE) $(NVCCFLAGS) -o $@ -c $<

clean:	
	rm -f *.o $(BINARIES)
