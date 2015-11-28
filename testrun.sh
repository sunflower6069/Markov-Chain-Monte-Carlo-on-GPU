#!/bin/bash
### This is a bash script to run the Markov Chain Monte Carlo Simulation on GPUs
### Please refer to the README file and pdf documents for more information 
### Xin Yan, Dec. 15, 2014

###############################################
#          General Usage Guidance             #
###############################################
# Command line input: $./MCMCpar <N> <m> <blockDim> <maxMCiter>
# <N>: total number of hard-sphere particles in the simulation system
# <m>: number of cells in each dimension of the simulation system.
#      The whole simulation system is decomposed into cells where 
#      each particle resides in a cell. Massive parallelization is 
#      achieved by assigning each cell to a single thread on GPU.
#      The current simulation protocol is designed to use m as multiples 
#      of 8, since the initial system is built from small box of 8x8 cells.
#      Ideally, N = m*m. 
# <blockDim>: number of threads in each dimension of a block. To make the
#       maximum use of the GPU, it is recommended that m/2 is multiples of 
#       blockDim.
# <maxMCiter>: total number of Monte Carlo sweep iterations. Each MC sweep 
#       will sweep over all cells in the sytem and perform trial moves 
#       multiple times to each partile. At the end of each MC sweep, a 
#       cell shift will be performed, which redraws the cell boundaries in 
#       a randomly chosen location.


# Testjobs
./MCMCpar 256 16 8 1 > test1.out
./MCMCpar 1024 32 8 5 > test2.out
./MCMCpar 1024 32 16 10 > test3.out

