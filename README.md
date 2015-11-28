### Markov Chain Monte Carlo Simulation on GPUs for hard-sphere particles ###
### Author: Xin Yan ###
### Date: Dec. 15, 2014 ###

#################################
#    General Introduction       #
#################################
This project aims at implementing a massively paralleled method on GPUs that performs 
Monte Carlo simulations of thermodynamic ensembles of hard-sphere particles that obeys
the detailed balance principle on a single Markov chain.

The algorithms used in the program is based on the following publication:
Anderson, J. A. et. al, Massively parallel Monte Carlo for many-particle simulations on GPUs. 
Journal of Computational Physics 2013, 254, 27.

The pdf of the paper can be found in the current directory.


#################################
# Instructions on Using MCMCpar #
#################################

This directory contains a Markov chain Monte Carlo simulation code in MCMCpar.cu.

Please make sure to load the required modules before running the program via command:

module load Langs/Intel/14 Langs/Cuda/6.0

To build the sample code, run:

make

This make command uses the makefile Makefile, which invokes the nvcc compiler 
to build the code. 

Once the code is built, you can execute it using:

./MCMCpar <N> <m> <blockDim> <maxMCiter>

where 

    <N>: total number of hard-sphere particles in the simulation system

    <m>: number of cells in each dimension of the simulation system.
    
    <blockDim>: number of threads in each dimension of a block.
    
    <maxMCiter>: total number of Monte Carlo sweep iterations. 

So this means that  

     blockDim.x = blockDim.y = <blockDim>
     blockDim.z = 1

For more specific instructions on how to set the parameters, please see FinalTestRun.sh

######################################
#         Testjobs                   #
######################################

After obtained the executable of the program, the userers are encouraged to run the testjobs:

bash FinalTestRun.sh

######################################
#           Output                   #
######################################

The output of the MCMCpar program is written to STDOUT which can be redirected to a file.

The output contains the information of the simulation system, execution status, as well as the 
wallclock time and particle coordinates.

