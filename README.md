# Markov-Chain-Monte-Carlo-on-GPU
This is a Markov chain Monte Carlo simulation implementation on hard-sphere-particle systems.

### Background ###
Monte Carlo algorithms are widely considered to be well suited to parallel computing [1]. However, in Monte Carlo simulations of thermodynamic ensembles of many-particle systems, the inherent sequential nature of statistical sampling in order to preserve the Markov chain precludes the possibility of using an “embarrassingly parallelizable” algorithm. This is because the generation of new configurations follows an irreducible Markov chain, in which every new configuration of the thermodynamic ensembles is dependent on the immediate predecessor, but not on any other previous steps [2].

With the mounting demand to simulate systems of greater size and longer time scales, massive parallelization of computer clusters becomes the natural choice to meet the requirements in scientific computing. In recent years, several parallel Markov chain Monte Carlo simulation techniques have been developed with Message Passing Interface (MPI) [3-4] and GPUs. [5] The parallelization could be achieved in different levels, and can be loosely classified into: 1) domain-decomposition; 2) parallel energy calculation; 3) hybrid Monte Carlo; and 4) task farming. [6] Among these techniques, the domain decomposition method divides the simulation system into smaller sub-domains, and assigns each worker with a single sub-domain that can be updated semi-independently. It is also the most popular technique thanks to its massive parallelizable potential and favorable scaling properties.

Herein we present a Markov chain Monte Carlo simulation implementation on hard-sphere-particle systems. The algorithms used in the program is largely depends on a recent work by Joshua Anderson et. al in ref. 5.

### Implementation ###
The GPU MCMCpar program was written in CUDA C and compiled with NVCC.

The .cu source code was compiled by each one of the following options:
nvcc –O3 -lcuda -lcudart –lm -gencode=arch=compute_20,code=\"sm_20,compute_20\"

And was linked by the command:
MCMCpar: MCMCpar.o
nvcc –O3 -lcuda -lcudart –lm gencode = arch = compute_20, code = \" sm_20, compute_20 \" -o $@ $^
.cu.o:
nvcc –O3 -lcuda -lcudart –lm gencode = arch = compute_20, code = \" sm_20, compute_20 \" -o $@ $<

### References ###
[1] Rosenthal, Jeffery S. Far East Journal of Theoretical Statistics 4 (2000), 207-236.
[2] Hammersley, J. M, Handscomb, D. C. Monte Carlo Methods. New York: John Wiley & Sons Inc. (1965), Chapter 9, 113-126.
[3] Ren, R., Orkoulas, G. Journal of Chemical Physics 126 (2007), 211102.
[4] O’Keeffe C. J., Orkoulas G. Journal of Chemical Physics 130 (2009), 134109.
[5] Anderson J. A., et. al, Journal of Computational Physics 254 (2013) 27-38.
